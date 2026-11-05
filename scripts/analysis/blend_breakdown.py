"""Single-fibre samples vs blends: exact-match, missed and added fibres, recall per fibre.

Input: a sample-level prediction file with columns sample, y_<class>, pred_<class>
(cv_sample_predictions.csv of a CV run). Uncertainty: bootstrap over samples.

Usage:
  python scripts/analysis/blend_breakdown.py <run_dir or csv> [--out file.txt]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ci(h, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    b = [h[rng.integers(0, len(h), len(h))].mean() for _ in range(n_boot)]
    return np.percentile(b, [2.5, 97.5]) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", help="run folder (uses cv_sample_predictions.csv) or a csv file")
    ap.add_argument("--out")
    a = ap.parse_args()
    p = Path(a.preds)
    p = p / "cv_sample_predictions.csv" if p.is_dir() else p
    s = pd.read_csv(p, index_col="sample")
    yc = [c for c in s.columns if c.startswith("y_")]
    classes = [c[2:] for c in yc]
    Y, P = s[yc].values.astype(int), s[["pred_" + c for c in classes]].values.astype(int)
    nf = Y.sum(1)
    hit = (Y == P).all(1)
    miss = ((Y == 1) & (P == 0)).any(1)
    extra = ((Y == 0) & (P == 1)).any(1)
    found = ((Y == 1) & (P == 1)).sum(1)

    lines = [f"{p}  ({len(s)} samples)", "",
             f"{'group':14s} {'n':>4s}  {'exact-match (95% CI)':24s} {'misses a fibre':>15s} {'adds a fibre':>13s}"]
    groups = [("single fibre", nf == 1), ("all blends", nf >= 2)] + \
             [(f"  {k} fibres", nf == k) for k in sorted(set(nf[nf >= 2]))]
    for name, m in groups:
        lo, hi = ci(hit[m])
        lines.append(f"{name:14s} {m.sum():4d}  {hit[m].mean()*100:5.1f}% ({lo:4.1f} to {hi:4.1f})      "
                     f"{miss[m].mean()*100:10.1f}% {extra[m].mean()*100:12.1f}%")
    b = nf >= 2
    lines += ["", f"blends: share of true fibres found {(found[b] / nf[b]).mean():.3f}, "
                  f"at least one fibre right {(found[b] > 0).mean():.3f}", "",
              f"{'fibre':12s} {'recall alone (n)':>18s} {'recall in blend (n)':>21s}"]
    for i, c in enumerate(classes):
        cells = []
        for m in (nf == 1, nf >= 2):
            pos = (Y[:, i] == 1) & m
            cells.append(f"{P[pos, i].mean():.2f} ({pos.sum()})" if pos.sum() else "-")
        lines.append(f"{c:12s} {cells[0]:>18s} {cells[1]:>21s}")
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
