"""Train and evaluate one configuration.

Examples
  python scripts/train.py --paths configs/paths_local.yaml configs/crop448_holdout.yaml
  python scripts/train.py --paths configs/paths_colab.yaml configs/resize448_holdout.yaml --set train.lr=1e-4
  # smoke test: 2 epochs, 3 batches each
  python scripts/train.py --paths configs/paths_local.yaml configs/crop448_holdout.yaml \
      --set train.epochs=2 --set train.max_batches=3 --set name=smoke --set out_dir=results/smoke
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baip.config import load_config  # noqa: E402
from baip.engine import run  # noqa: E402

if __name__ == "__main__":   # guard needed: macOS data-loader workers re-import this file
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", help="experiment yaml in configs/")
    ap.add_argument("--paths", required=True, help="machine-specific paths yaml")
    ap.add_argument("--set", action="append", default=[], metavar="key=value", help="override, e.g. train.lr=1e-4")
    a = ap.parse_args()
    run(load_config([a.paths, a.experiment], a.set))
