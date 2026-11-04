"""Run k-fold cross-validation for one configuration, then pool the folds.

Folds already finished in the group folder are skipped, so after a Colab disconnect you
simply run the same command again. All experiments use the same folds (split.split_seed),
so their results can be compared sample by sample (McNemar).

Examples
  # all 5 folds one after another (Colab, one GPU)
  python scripts/run_cv.py configs/crop448_cv.yaml --paths configs/paths_colab.yaml
  # only folds 0 and 1 (e.g. split over two sessions or jobs), into the same group folder
  python scripts/run_cv.py configs/crop448_cv.yaml --paths configs/paths_colab.yaml --folds 0 1
  # pool whatever folds are finished, without training
  python scripts/run_cv.py configs/crop448_cv.yaml --paths configs/paths_colab.yaml --aggregate-only
"""
import argparse
import copy
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baip.config import load_config  # noqa: E402
from baip.cv import aggregate  # noqa: E402
from baip.engine import run  # noqa: E402

# settings that change results; paths, worker counts and memory knobs are ignored
IGNORE = {"data": {"image_dirs", "labels_xlsx", "cache_dir", "num_workers"},
          "split": {"fold"}, "train": {"eval_batch_tiles"}}


def result_settings(cfg):
    out = {"seed": cfg["seed"], "model": cfg["model"]}
    for sec, skip in IGNORE.items():
        out[sec] = {k: v for k, v in cfg[sec].items() if k not in skip}
    return out


def check_same(fold_dir, cfg):
    old = yaml.safe_load((fold_dir / "config.yaml").read_text())
    a, b = result_settings(old), result_settings(cfg)
    if a != b:
        diff = [f"{sec}.{k}: {a[sec].get(k)} -> {b[sec].get(k)}" for sec in ("data", "split", "train")
                for k in set(a[sec]) | set(b[sec]) if a[sec].get(k) != b[sec].get(k)]
        diff += [k for k in ("seed", "model") if a[k] != b[k]]
        sys.exit(f"{fold_dir} was trained with other settings ({'; '.join(diff)}).\n"
                 f"Use another --group, or delete that folder to retrain.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--paths", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="key=value")
    ap.add_argument("--folds", type=int, nargs="*", help="default: all folds")
    ap.add_argument("--group", help="folder name under out_dir (default: <name>); reuse it to resume")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config([a.paths, a.experiment], a.set)
    cfg["split"]["scheme"] = "cv"
    k = cfg["split"]["n_folds"]
    # default group name includes the overrides, so a variant (e.g. train.lr=1e-4) never
    # resumes into the folder of the base experiment
    suffix = "".join("__" + re.sub(r"[^A-Za-z0-9.=-]", "_", o) for o in a.set
                     if not o.split("=")[0] in ("out_dir", "ckpt_dir", "data.num_workers", "train.eval_batch_tiles"))
    group = Path(cfg["out_dir"]) / (a.group or cfg["name"] + suffix)
    folds = a.folds if a.folds else list(range(k))

    if not a.aggregate_only:
        for f in folds:
            fold_dir = group / f"fold{f}"
            if (fold_dir / "metrics.json").exists():
                check_same(fold_dir, cfg)
                print(f"fold {f}: already finished, skipped ({fold_dir})")
                continue
            print(f"\n######## {group.name}  fold {f} of {k}")
            c = copy.deepcopy(cfg)
            c["split"]["fold"] = f
            c["run_dir"] = str(fold_dir)
            c["name"] = f"{cfg['name']}_fold{f}"
            run(c)
    print()
    aggregate(group, n_folds=k)
