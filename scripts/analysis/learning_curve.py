"""Learning curve and power-law slope from the learning-curve runs.

Finds, in <results_dir>, the full runs (<mode>448_cv_v1, or <mode>448_cv) and the reduced
runs (<mode>448_lc_fXXX[_sN]) for mode in crop and resize. For each run: mean number of
training samples per fold (from foldK/split.json) and the pooled test sample exact-match.

Fit per mode (Hestness et al. 2017): error = a * n_train^(-alpha), as a straight line in
log-log space through all runs of that mode. The 95% CI of alpha comes from a paired
bootstrap over test samples: every run is scored on the same 736 samples, so one resampling
of sample IDs is applied to all runs at once.

Then extrapolates to a larger training set (default: as if 736 + 1354 samples were available
with the same 65/15/20 split, i.e. about 2.84 times the current training set).

Usage:
  python scripts/analysis/learning_curve.py <results_dir> [--extra-samples 1354] [--out <dir>]
Writes learning_curve.csv, learning_curve_fit.txt and learning_curve.png to --out
(default: <results_dir>/analyses/learning_curve).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PAT = re.compile(r"^(crop|resize)448_(?:cv(?:_v1)?|lc_f(\d{3})(?:_s(\d+))?)$")


def runs(results):
    rows = []
    for d in sorted(Path(results).iterdir()):
        m = PAT.match(d.name)
        if not m or not (d / "cv_sample_predictions.csv").exists():
            continue
        folds = sorted(d.glob("fold*/split.json"))
        if len(folds) < 5:
            print(f"skip {d.name}: {len(folds)} of 5 folds")
            continue
        n_tr = np.mean([len(json.loads(f.read_text())["train"]) for f in folds])
        s = pd.read_csv(d / "cv_sample_predictions.csv", index_col="sample").sort_index()
        yc = [c for c in s.columns if c.startswith("y_")]
        pc = ["pred_" + c[2:] for c in yc]
        hit = (s[yc].values == s[pc].values).all(1)
        e = yc.index("y_Elastane")
        pos = s[yc[e]].values == 1
        rows.append({"run": d.name, "mode": m.group(1),
                     "frac": int(m.group(2)) / 100 if m.group(2) else 1.0,
                     "subsample_seed": int(m.group(3) or 0) if m.group(2) else None,
                     "n_train": n_tr, "n_test": len(s), "sample_EM": hit.mean(),
                     "elastane_recall": (s[pc[e]].values[pos] == 1).mean(), "_hit": hit, "_ids": s.index})
    return rows


def fit(n, err):
    """OLS of log(err) on log(n). Returns (a, alpha)."""
    b, c = np.polyfit(np.log(n), np.log(err), 1)
    return float(np.exp(c)), float(-b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--extra-samples", type=int, default=1354)
    ap.add_argument("--out")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.results) / "analyses" / "learning_curve"
    out.mkdir(parents=True, exist_ok=True)

    R = runs(a.results)
    if not R:
        raise SystemExit("no complete runs found")
    ids0 = R[0]["_ids"]
    assert all((r["_ids"] == ids0).all() for r in R), "runs were not scored on the same test samples"
    table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in R])
    table.to_csv(out / "learning_curve.csv", index=False)

    lines, fits = [], {}
    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, len(ids0), len(ids0)) for _ in range(a.n_boot)]
    for mode in ("crop", "resize"):
        rs = [r for r in R if r["mode"] == mode]
        if len(rs) < 3:
            lines.append(f"{mode}: only {len(rs)} complete runs, need at least 3 for a fit")
            continue
        n = np.array([r["n_train"] for r in rs])
        err = np.array([1 - r["sample_EM"] for r in rs])
        a_, alpha = fit(n, err)
        n_full = max(r["n_train"] for r in rs if r["frac"] == 1.0) if any(r["frac"] == 1.0 for r in rs) else n.max()
        factor = (rs[0]["n_test"] + a.extra_samples) / rs[0]["n_test"]
        n_new = n_full * factor
        em_new = 1 - a_ * n_new ** -alpha
        alphas, em_b = [], []
        for idx in boot_idx:
            e_b = np.array([1 - r["_hit"][idx].mean() for r in rs])
            if (e_b > 0).all():
                ab, alb = fit(n, e_b)
                alphas.append(alb)
                em_b.append(1 - ab * n_new ** -alb)
        lo, hi = np.percentile(alphas, [2.5, 97.5])
        em_new_ci = np.percentile(em_b, [2.5, 97.5])
        fits[mode] = (a_, alpha, n_new)
        lines += [f"{mode}: {len(rs)} runs, n_train {n.min():.0f} to {n.max():.0f}",
                  f"  fit: error = {a_:.3f} * n_train^(-{alpha:.3f})",
                  f"  alpha = {alpha:.3f} (paired bootstrap 95% CI {lo:.3f} to {hi:.3f}; literature range 0.07 to 0.35)",
                  f"  extrapolation to n_train = {n_new:.0f} ({factor:.2f} x, i.e. +{a.extra_samples} samples, same split):",
                  f"    sample exact-match about {em_new*100:.1f}% (paired bootstrap 95% CI {em_new_ci[0]*100:.1f} to {em_new_ci[1]*100:.1f}%;"
                  f" covers test-sample noise only, not model choice or domain change)",
                  f"  caveat: same domain (7.58 um/px) assumed; the new microscope is a different domain, so use the slope, not the level."]
    for r in R:
        lines.append(f"  {r['run']:24s} frac {r['frac']:.2f}  n_train {r['n_train']:6.1f}  sample EM {r['sample_EM']*100:5.1f}%"
                     f"  elastane recall {r['elastane_recall']:.2f}")
    text = "\n".join(lines)
    (out / "learning_curve_fit.txt").write_text(text + "\n")
    print(text)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"crop": "#2a78d6", "resize": "#eb6834"}
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for mode in ("crop", "resize"):
        rs = [r for r in R if r["mode"] == mode]
        if not rs:
            continue
        ax.plot([r["n_train"] for r in rs], [(1 - r["sample_EM"]) * 100 for r in rs], "o" if mode == "crop" else "s",
                color=col[mode], ms=7, label=f"{mode} 448 (measured)")
        if mode in fits:
            a_, alpha, n_new = fits[mode]
            xs = np.geomspace(min(r["n_train"] for r in rs), n_new, 50)
            ax.plot(xs, a_ * xs ** -alpha * 100, color=col[mode], lw=2, ls="-", alpha=0.8,
                    label=f"{mode} fit, alpha = {alpha:.2f}")
            ax.axvline(n_new, color="#52514e", lw=1, ls=(0, (3, 3)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training samples per fold (log)"); ax.set_ylabel("sample exact-match error, % (log)")
    ax.set_title("Learning curve, 5-fold CV, pooled test (n = 736); dashed = extrapolation target",
                 loc="left", fontsize=10)
    ax.grid(True, which="both", color="#e6e5e1", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=160)
    print("saved", out)


if __name__ == "__main__":
    main()
