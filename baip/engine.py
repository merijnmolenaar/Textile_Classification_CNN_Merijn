"""Training loop, prediction and run logging."""
import csv
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data as D
from . import metrics as M
from .config import dump_config
from .model import Lamb, TextileCNN
from .splits import describe, get_split


# ---------------------------------------------------------------- environment

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(wid):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def git_info():
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=here, text=True).strip())
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def env_info(device, cfg=None):
    info = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
            "host": platform.node(), **git_info()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    info["amp"] = str(amp_dtype(device, cfg)) if cfg else None
    return info


# ---------------------------------------------------------------- core loops

def _collate(batch):
    xs, ys, sids, paths = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(sids), list(paths)


def make_loader(recs, cfg, train, seed):
    ds = D.ImageDataset(recs, cfg, train)
    bs = cfg["train"]["batch_size"]
    if not train:   # cap the number of tiles per forward pass at evaluation
        bs = max(1, cfg["train"]["eval_batch_tiles"] // ds.views)
    nw = cfg["data"]["num_workers"]
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=nw, collate_fn=_collate,
                      pin_memory=torch.cuda.is_available(), persistent_workers=nw > 0,
                      worker_init_fn=_worker_init, generator=g)


def amp_dtype(device, cfg):
    """Mixed precision on CUDA only: bf16 on A100/L4/H100 and newer (compute capability 8+),
    fp16 with loss scaling on older GPUs such as the free-tier T4, which has no bf16."""
    if not (cfg["train"]["amp"] and device.type == "cuda"):
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def _autocast(device, cfg):
    dt = amp_dtype(device, cfg)
    return torch.autocast("cuda", dtype=dt) if dt else torch.autocast("cpu", enabled=False)


def train_epoch(model, loader, opt, crit, device, cfg, scaler=None):
    model.train()
    tot, n, hits = 0.0, 0, 0
    t = cfg["train"]["threshold"]
    maxb = cfg["train"]["max_batches"]
    for b, (x, y, _, _) in enumerate(loader):
        if maxb and b >= maxb:
            break
        x = x.flatten(0, 1).to(device, non_blocking=True)    # training has V = 1
        y = y.to(device, non_blocking=True)
        with _autocast(device, cfg):
            logits = model(x)
        loss = crit(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:   # fp16: scale the loss so small gradients do not underflow
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        tot += loss.item() * len(y)
        n += len(y)
        hits += ((torch.sigmoid(logits.float()) > t) == (y > 0.5)).all(1).sum().item()
    return tot / max(n, 1), hits / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, cfg, crit=None):
    """Image-level probabilities (mean over views). Returns dict(probs, labels, sids, paths, loss)."""
    model.eval()
    P, Y, S, F, tot = [], [], [], [], 0.0
    maxb = cfg["train"]["max_batches"]
    for b, (x, y, sids, paths) in enumerate(loader):
        if maxb and b >= maxb:
            break
        B, V = x.shape[:2]
        with _autocast(device, cfg):
            logits = model(x.flatten(0, 1).to(device, non_blocking=True)).float()
        prob = torch.sigmoid(logits).view(B, V, -1).mean(1)
        if crit is not None:
            pc = prob.clamp(1e-6, 1 - 1e-6)
            tot += nn.functional.binary_cross_entropy(pc, y.to(device), reduction="sum").item() / y.shape[1]
        P.append(prob.cpu().numpy())
        Y.append(y.numpy())
        S += sids
        F += paths
    probs, labels = np.vstack(P), np.vstack(Y)
    return {"probs": probs, "labels": labels, "sids": S, "paths": F, "loss": tot / max(len(S), 1)}


def save_predictions(pred, classes, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "image", *[f"p_{c}" for c in classes], *[f"y_{c}" for c in classes]])
        for s, f, p, y in zip(pred["sids"], pred["paths"], pred["probs"], pred["labels"]):
            w.writerow([s, os.path.basename(f), *np.round(p, 5), *y.astype(int)])


# ---------------------------------------------------------------- one full run

def run(cfg):
    seed_all(cfg["seed"])
    device = pick_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    cfg = D.cache_locally(cfg)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg.get("run_dir"):   # set by scripts/run_cv.py: <group>/fold<k>
        run_dir = Path(cfg["run_dir"])
        ckpt = Path(cfg["ckpt_dir"]) / f"{run_dir.parent.name}_{run_dir.name}.pt"
    else:
        run_dir = Path(cfg["out_dir"]) / f"{stamp}_{cfg['name']}"
        ckpt = Path(cfg["ckpt_dir"]) / f"{stamp}_{cfg['name']}.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    env = env_info(device, cfg)
    print(f"run {run_dir.name} | device {env['device']} {env.get('gpu', '')} | commit {str(env['commit'])[:8]}"
          + (" (uncommitted changes!)" if env["dirty"] else ""))

    samples, classes = D.build_index(cfg)
    with open(run_dir / "labels.csv", "w", newline="") as fh:   # exact labels this run used
        w = csv.writer(fh)
        w.writerow(["sample", *classes, "n_images"])
        w.writerows([[sid, *y.astype(int), len(paths)] for sid, y, paths in samples])
    split = get_split(samples, cfg["split"])
    table, counts = describe(split, samples, classes)
    print(table)
    dump_config(cfg, run_dir / "config.yaml")
    (run_dir / "split.json").write_text(json.dumps({**split, "class_counts": counts}, indent=1))
    (run_dir / "env.json").write_text(json.dumps(env, indent=1))

    tr = make_loader(D.records(samples, split["train"]), cfg, True, cfg["seed"])
    va = make_loader(D.records(samples, split["val"]), cfg, False, cfg["seed"])
    te = make_loader(D.records(samples, split["test"]), cfg, False, cfg["seed"]) if split["test"] else None

    tc = cfg["train"]
    model = TextileCNN(cfg["model"]["backbone"], len(classes), cfg["model"]["pretrained"]).to(device)
    pw = None
    if tc["pos_weight"]:
        Ytr = np.stack([y for sid, y, _ in samples if sid in set(split["train"])])
        pw = torch.tensor(np.clip((len(Ytr) - Ytr.sum(0)) / np.maximum(Ytr.sum(0), 1), 1, 50),
                          dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = Lamb(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"], eta_min=tc["eta_min"])

    scaler = torch.amp.GradScaler("cuda") if amp_dtype(device, cfg) == torch.float16 else None
    hist, best, best_ep, since = [], -1.0, 0, 0
    t_start = time.time()
    print(f"\n{'ep':>3} {'lr':>9} {'trLoss':>7} {'trAcc':>6} {'vaLoss':>7} {'vaImg':>6} {'vaSmp':>6} {'sec':>5}")
    for ep in range(1, tc["epochs"] + 1):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        tl, ta = train_epoch(model, tr, opt, crit, device, cfg, scaler)
        vp = predict(model, va, device, cfg, crit)
        sched.step()
        ev = M.evaluate(vp, classes, tc["threshold"])
        vi, vs = ev["image"]["exact_match"], ev["sample"]["exact_match"]
        score = (vi + 1e-3 * ev["image"]["micro_f1"]) if tc["select_on"] == "val_image_acc" \
            else (vs + 1e-3 * ev["sample"]["macro_f1"])
        sec = time.time() - t0
        flag = ""
        if score > best:
            best, best_ep, since, flag = score, ep, 0, " *"
            torch.save({"model_state": model.state_dict(), "classes": classes, "epoch": ep, "config": cfg}, ckpt)
        else:
            since += 1
        hist.append({"epoch": ep, "lr": lr, "train_loss": tl, "train_img_acc": ta, "val_loss": vp["loss"],
                     "val_img_acc": vi, "val_smp_acc": vs, "val_smp_macro_f1": ev["sample"]["macro_f1"], "sec": sec})
        print(f"{ep:>3} {lr:>9.2e} {tl:>7.4f} {ta*100:>6.2f} {vp['loss']:>7.4f} {vi*100:>6.2f} {vs*100:>6.2f} {sec:>5.0f}{flag}")
        if ep == 1:
            print(f"    first epoch {sec:.0f} s -> up to {sec * tc['epochs'] / 60:.0f} min for {tc['epochs']} epochs")
        with open(run_dir / "history.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hist[0]))
            w.writeheader()
            w.writerows(hist)
        if since >= tc["patience"]:
            print(f"early stop: no improvement for {tc['patience']} epochs")
            break

    # ---- final evaluation with the best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model_state"])
    vp = predict(model, va, device, cfg)
    save_predictions(vp, classes, run_dir / "preds_val.csv")
    _, vsp, vsl = M.to_samples(vp["probs"], vp["labels"].astype(int), vp["sids"])
    thr = M.tune_threshold(vsp, vsl) if tc["tune_threshold"] else tc["threshold"]
    out = {"name": cfg["name"], "run": run_dir.name, "best_epoch": best_ep, "epochs_run": len(hist),
           "minutes": (time.time() - t_start) / 60, "classes": classes, "um_per_px": cfg["data"]["um_per_px"],
           "mode": cfg["data"]["mode"], "split": cfg["split"]["scheme"],
           "val_at_0.5": M.evaluate(vp, classes, 0.5), "val_threshold": thr}
    if te is not None:
        tp = predict(model, te, device, cfg)
        save_predictions(tp, classes, run_dir / "preds_test.csv")
        out["test_at_0.5"] = M.evaluate(tp, classes, 0.5)
        out["test_at_val_threshold"] = M.evaluate(tp, classes, thr)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=1))

    # one summary line per run, appended to results/runs.csv
    head = "test_at_val_threshold" if te is not None else "val_at_0.5"
    s = out[head]["sample"]
    row = {"run": (f"{run_dir.parent.name}/{run_dir.name}" if cfg.get("run_dir") else run_dir.name), "name": cfg["name"], "commit": str(env["commit"])[:8], "dirty": env["dirty"],
           "device": env.get("gpu", env["device"]), "mode": cfg["data"]["mode"], "split": cfg["split"]["scheme"],
           "fold": cfg["split"]["fold"] if cfg["split"]["scheme"] == "cv" else None, "lr": tc["lr"], "augment": cfg["data"]["augment"],
           "reported_on": head, "threshold": out[head]["threshold"], "best_epoch": best_ep,
           "sample_acc": round(s["exact_match"], 4), "sample_acc_ci95": [round(v, 4) for v in s["exact_match_ci95"]],
           "image_acc": round(out[head]["image"]["exact_match"], 4), "sample_macro_f1": round(s["macro_f1"], 4),
           "elastane_recall": s["per_class"].get("Elastane", {}).get("recall"),
           "minutes": round(out["minutes"], 1)}
    summ = Path(cfg["out_dir"]) / "runs.csv"
    new = not summ.exists()
    with open(summ, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)

    print(f"\n== {head}: sample exact-match {s['exact_match']*100:.2f}% "
          f"(95% CI {s['exact_match_ci95'][0]*100:.1f} to {s['exact_match_ci95'][1]*100:.1f}), "
          f"image {out[head]['image']['exact_match']*100:.2f}%, macro-F1 {s['macro_f1']:.3f}, "
          f"threshold {out[head]['threshold']}")
    print(f"results in {run_dir}")
    return out
