"""Pool the test predictions of k cross-validation folds into one result.

Every sample is in the test set of exactly one fold, so pooling gives one prediction per
sample for the whole dataset. Each fold's predictions are thresholded with the threshold
that fold chose on its own validation set (never on test data).
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M


def _fold_dirs(group):
    return sorted(d for d in Path(group).glob("fold*") if (d / "metrics.json").exists())


def aggregate(group, n_folds=None):
    group = Path(group)
    folds = _fold_dirs(group)
    if not folds:
        raise FileNotFoundError(f"no finished folds in {group}")
    if n_folds and len(folds) < n_folds:
        print(f"WARNING: only {len(folds)} of {n_folds} folds finished; result covers part of the data")

    per_fold, commits, frames = [], set(), []
    classes = None
    for d in folds:
        m = json.loads((d / "metrics.json").read_text())
        env = json.loads((d / "env.json").read_text())
        commits.add((env.get("commit"), env.get("dirty")))
        classes = classes or m["classes"]
        if m["classes"] != classes:
            raise ValueError(f"{d.name} has different classes")
        df = pd.read_csv(d / "preds_test.csv")
        df["fold"] = d.name
        df["threshold"] = m["val_threshold"]
        frames.append(df)
        t = m["test_at_val_threshold"]["sample"]
        per_fold.append({"fold": d.name, "n": t["n"], "threshold": m["val_threshold"], "best_epoch": m["best_epoch"],
                         "sample_acc": t["exact_match"], "sample_macro_f1": t["macro_f1"],
                         "image_acc": m["test_at_val_threshold"]["image"]["exact_match"],
                         "elastane_recall": t["per_class"].get("Elastane", {}).get("recall"),
                         "minutes": m["minutes"]})
    if len(commits) > 1:
        print(f"WARNING: folds ran on different code versions: {commits}")

    df = pd.concat(frames, ignore_index=True)
    dup = df.groupby("sample")["fold"].nunique()
    if (dup > 1).any():
        raise ValueError(f"samples tested in more than one fold: {list(dup[dup > 1].index)[:5]}")

    pc = [f"p_{c}" for c in classes]
    yc = [f"y_{c}" for c in classes]
    # image level: threshold each image with its fold's threshold
    Pi = (df[pc].to_numpy() > df[["threshold"]].to_numpy()).astype(int)
    Li = df[yc].to_numpy().astype(int)
    # sample level: mean probability over the sample's images, then its fold's threshold
    s = df.groupby("sample").agg({**{c: "mean" for c in pc}, **{c: "first" for c in yc}, "threshold": "first", "fold": "first"})
    Ps = (s[pc].to_numpy() > s[["threshold"]].to_numpy()).astype(int)
    Ls = s[yc].to_numpy().astype(int)
    Ps05 = (s[pc].to_numpy() > 0.5).astype(int)

    accs = [f["sample_acc"] for f in per_fold]
    out = {
        "group": group.name, "folds": [f["fold"] for f in per_fold], "classes": classes,
        "commits": sorted(str(c) for c in commits),
        "pooled_sample": M.report(Ls, Ps, classes),
        "pooled_image": M.report(Li, Pi, classes),
        "pooled_sample_at_0.5": M.report(Ls, Ps05, classes),
        "fold_sample_acc_mean": float(np.mean(accs)), "fold_sample_acc_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
        "per_fold": per_fold,
    }
    (group / "cv_metrics.json").write_text(json.dumps(out, indent=1))
    s.assign(**{f"pred_{c}": Ps[:, i] for i, c in enumerate(classes)}).to_csv(group / "cv_sample_predictions.csv")

    ps = out["pooled_sample"]
    lines = [f"{group.name}: {len(per_fold)} folds, {ps['n']} test samples (each tested once)",
             f"  sample exact-match  {ps['exact_match']*100:.2f}%  (95% CI {ps['exact_match_ci95'][0]*100:.1f} to {ps['exact_match_ci95'][1]*100:.1f})",
             f"  per fold            {', '.join(f'{a*100:.1f}' for a in accs)}  (mean {np.mean(accs)*100:.1f})",
             f"  image exact-match   {out['pooled_image']['exact_match']*100:.2f}%",
             f"  sample macro-F1     {ps['macro_f1']:.3f}",
             f"  {'fibre':<11} {'prec':>5} {'recall':>6} {'n':>4}"]
    for c in classes:
        r = ps["per_class"][c]
        lines.append(f"  {c:<11} {r['precision']:>5.2f} {r['recall']:>6.2f} {r['support']:>4}")
    text = "\n".join(lines)
    (group / "cv_summary.txt").write_text(text + "\n")
    print(text)

    # one line for the whole CV in runs.csv, next to the per-fold lines
    summ = group.parent / "runs.csv"
    row = {"run": f"{group.name}/pooled", "name": group.name, "commit": "/".join(str(c[0])[:8] for c in commits),
           "dirty": any(c[1] for c in commits), "device": "", "mode": "", "split": "cv", "fold": "pooled",
           "lr": "", "augment": "", "reported_on": f"pooled test, {len(per_fold)} folds", "threshold": "per fold",
           "best_epoch": "", "sample_acc": round(ps["exact_match"], 4),
           "sample_acc_ci95": [round(v, 4) for v in ps["exact_match_ci95"]],
           "image_acc": round(out["pooled_image"]["exact_match"], 4), "sample_macro_f1": round(ps["macro_f1"], 4),
           "elastane_recall": ps["per_class"].get("Elastane", {}).get("recall"),
           "minutes": round(sum(f["minutes"] for f in per_fold), 1)}
    if n_folds and len(per_fold) < n_folds:
        return out   # partial result: not logged to runs.csv
    new = not summ.exists()
    with open(summ, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return out
