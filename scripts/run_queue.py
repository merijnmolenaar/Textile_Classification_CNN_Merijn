"""Run a list of experiments one after another (e.g. overnight on Colab).

Each line of the queue file is an experiment config (blank lines and # comments ignored).
CV configs (split.scheme: cv) run as a `scripts/run_cv.py` process; single-split configs as a
`scripts/train.py` process writing to <out_dir>/holdout/<name>/. So:
  - a failing experiment is logged and the queue moves on to the next one;
  - finished folds and finished single runs are skipped, so after a disconnect you start the
    same queue again and it continues where it stopped (an interrupted run restarts from epoch 1).

Output, all under <out_dir> from the paths file:
  queue_logs/<experiment>.log   full console output of that experiment (unbuffered)
  queue_logs/queue_status.txt   one line per experiment: start, end, minutes, OK / FAILED (exit code)

Example (Colab):
  python scripts/run_queue.py configs/learning_curve/queue.txt --paths configs/paths_colab.yaml
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_queue(path):
    lines = [ln.split("#", 1)[0].strip() for ln in Path(path).read_text().splitlines()]
    return [ln for ln in lines if ln]


def command(exp, paths, out_dir):
    """CV configs (split.scheme: cv) run through run_cv.py, which skips finished folds.
    Other configs (one fixed split) run through train.py into <out_dir>/holdout/<name>/;
    returns None if that run already has metrics.json."""
    cfg = yaml.safe_load((ROOT / exp).read_text()) or {}
    if (cfg.get("split") or {}).get("scheme") == "cv":
        return [sys.executable, "-u", str(ROOT / "scripts/run_cv.py"), str(ROOT / exp), "--paths", paths]
    run_dir = out_dir / "holdout" / cfg.get("name", Path(exp).stem)
    if (run_dir / "metrics.json").exists():
        return None
    return [sys.executable, "-u", str(ROOT / "scripts/train.py"), str(ROOT / exp), "--paths", paths,
            "--set", f"run_dir={run_dir}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue", help="text file with one experiment config per line")
    ap.add_argument("--paths", required=True)
    ap.add_argument("--dry-run", action="store_true", help="only list what would run")
    a = ap.parse_args()

    exps = read_queue(a.queue)
    missing = [e for e in exps if not (ROOT / e).exists()]
    if missing:
        sys.exit(f"config files not found: {missing}")
    out_dir = Path(yaml.safe_load(Path(a.paths).read_text())["out_dir"])
    log_dir = out_dir / "queue_logs"
    status = log_dir / "queue_status.txt"

    print(f"queue: {len(exps)} experiments, logs in {log_dir}")
    for i, e in enumerate(exps, 1):
        print(f"  {i}. {e}")
    if a.dry_run:
        return
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(status, "a") as fh:
        fh.write(f"\n=== queue {Path(a.queue).name} started {datetime.now():%Y-%m-%d %H:%M} ({len(exps)} experiments)\n")
    n_ok = 0
    for i, e in enumerate(exps, 1):
        name = Path(e).stem
        log = log_dir / f"{name}.log"
        t0 = time.time()
        start = datetime.now()
        print(f"\n[{i}/{len(exps)}] {name}  started {start:%H:%M}  -> {log}", flush=True)
        cmd = command(e, a.paths, out_dir)
        if cmd is None:
            print(f"    already finished, skipped", flush=True)
            with open(status, "a") as fh:
                fh.write(f"{name:28s} skipped (already finished)\n")
            n_ok += 1
            continue
        with open(log, "a") as fh:
            fh.write(f"\n===== {start:%Y-%m-%d %H:%M:%S} {e}\n")
            fh.flush()
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
        mins = (time.time() - t0) / 60
        verdict = "OK" if rc == 0 else f"FAILED (exit {rc})"
        n_ok += rc == 0
        line = f"{name:28s} {start:%H:%M} -> {datetime.now():%H:%M}  {mins:6.1f} min  {verdict}"
        print("   ", line, flush=True)
        with open(status, "a") as fh:
            fh.write(line + "\n")
    with open(status, "a") as fh:
        fh.write(f"=== queue finished {datetime.now():%Y-%m-%d %H:%M}: {n_ok}/{len(exps)} OK\n")
    print(f"\nqueue finished: {n_ok}/{len(exps)} OK. Status: {status}")


if __name__ == "__main__":
    main()
