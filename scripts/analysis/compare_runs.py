"""Paired comparison of two 5-fold CV runs on identical folds (e.g. crop448_cv_v1 vs resize448_cv).

Usage: python compare_runs.py <runA_dir> <runB_dir> [out.csv]
- checks that every fold has identical train/val/test sample IDs
- sample level: average image probabilities per sample, threshold = that fold's own val_threshold
- pooled over all test samples: exact-match (bootstrap CI), image exact-match, micro/macro F1
  (macro = per-fibre F1 from pooled counts, then mean; Forman & Scholz 2010), per-fibre P/R
- paired: exact McNemar on sample exact-match, on elastane positives (recall) and on
  elastane negatives (false alarms); paired bootstrap CI of the exact-match difference
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

A, B = Path(sys.argv[1]), Path(sys.argv[2])
CLASSES = ["Cotton", "Wool", "Hemp", "Linen", "Elastane", "Polyester", "Viscose", "Silk",
           "Tencel", "PLA", "Polyamide", "Alpaca", "Peat fibre"]
PC, YC = [f"p_{c}" for c in CLASSES], [f"y_{c}" for c in CLASSES]


def load(run):
    S, Y, I, IY, thr, info = [], [], [], [], {}, {}
    for k in range(5):
        f = run / f"fold{k}"
        m = json.loads((f / "metrics.json").read_text())
        t = m["val_threshold"]; thr[k] = t
        info[k] = (m["best_epoch"], m["epochs_run"], round(m["minutes"], 1),
                   m["val_at_0.5"]["sample"]["exact_match"])
        d = pd.read_csv(f / "preds_test.csv")
        I.append((d[PC].values > t).astype(int)); IY.append(d[YC].values.astype(int))
        g = d.groupby("sample")
        P = (g[PC].mean() > t).astype(int); P.columns = CLASSES
        L = g[YC].first().astype(int); L.columns = CLASSES
        S.append(P.assign(fold=k)); Y.append(L)
    return pd.concat(S), pd.concat(Y), np.vstack(I), np.vstack(IY), thr, info


for k in range(5):
    sa = json.loads((A / f"fold{k}/split.json").read_text())
    sb = json.loads((B / f"fold{k}/split.json").read_text())
    assert all(sorted(sa[p]) == sorted(sb[p]) for p in ("train", "val", "test")), f"fold {k} splits differ"
print("splits: identical in all 5 folds")

PA, YA, IA, IYA, tA, infA = load(A)
PB, YB, IB, IYB, tB, infB = load(B)
PB, YB = PB.loc[PA.index], YB.loc[PA.index]
assert (YA.values == YB.values).all()
L = YA[CLASSES].values
hA = (PA[CLASSES].values == L).all(1); hB = (PB[CLASSES].values == L).all(1)
n = len(L)


def boot(h, n_boot=5000, seed=0):
    r = np.random.default_rng(seed)
    return np.percentile([h[r.integers(0, len(h), len(h))].mean() for _ in range(n_boot)], [2.5, 97.5])


def f1s(P, L):
    tp = ((P == 1) & (L == 1)).sum(0); fp = ((P == 1) & (L == 0)).sum(0); fn = ((P == 0) & (L == 1)).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0); rec = np.where(tp + fn > 0, tp / (tp + fn), 0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0)
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum())
    return prec, rec, f1, micro, tp + fn


def mcnemar(a, b):
    """a, b: boolean 'correct' arrays. b_ = A right & B wrong, c_ = A wrong & B right."""
    b_, c_ = int((a & ~b).sum()), int((~a & b).sum())
    return b_, c_, (binomtest(b_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0)


rows = []
for name, P, I, IY, h in [(A.name, PA, IA, IYA, hA), (B.name, PB, IB, IYB, hB)]:
    pr, rc, f1, mi, sup = f1s(P[CLASSES].values, L)
    e = CLASSES.index("Elastane")
    lo, hi = boot(h)
    rows.append({"run": name, "n_samples": n, "sample_EM": h.mean(), "ci_lo": lo, "ci_hi": hi,
                 "image_EM": (I == IY).all(1).mean(), "micro_F1": mi, "macro_F1_13": f1.mean(),
                 "macro_F1_present_ge20": f1[sup >= 20].mean(),
                 "elastane_prec": pr[e], "elastane_rec": rc[e],
                 **{f"F1_{c}": v for c, v in zip(CLASSES, f1)},
                 **{f"rec_{c}": v for c, v in zip(CLASSES, rc)},
                 **{f"prec_{c}": v for c, v in zip(CLASSES, pr)}})
out = pd.DataFrame(rows).set_index("run")
sup = L.sum(0)

# paired tests
e = CLASSES.index("Elastane")
pos, neg = L[:, e] == 1, L[:, e] == 0
okA_e, okB_e = PA["Elastane"].values == L[:, e], PB["Elastane"].values == L[:, e]
tests = {
    "sample exact-match (all)": mcnemar(hA, hB),
    "elastane found (positives only)": mcnemar(okA_e[pos], okB_e[pos]),
    "elastane no false alarm (negatives only)": mcnemar(okA_e[neg], okB_e[neg]),
}
r = np.random.default_rng(1)
diffs = [(hB[i].mean() - hA[i].mean()) for i in (r.integers(0, n, n) for _ in range(5000))]
dlo, dhi = np.percentile(diffs, [2.5, 97.5])

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 50)
print("\n== pooled test, sample level (n = %d), per-fold val thresholds" % n)
print(out[["sample_EM", "ci_lo", "ci_hi", "image_EM", "micro_F1", "macro_F1_13", "macro_F1_present_ge20",
           "elastane_prec", "elastane_rec"]].round(4).T.to_string())
print("\n== per fibre (n = positives over all test folds): precision / recall / F1")
for i, c in enumerate(CLASSES):
    a, b = out.iloc[0], out.iloc[1]
    print(f"{c:11s} n={sup[i]:3d} | {A.name}: {a[f'prec_{c}']:.2f} {a[f'rec_{c}']:.2f} {a[f'F1_{c}']:.2f}"
          f" | {B.name}: {b[f'prec_{c}']:.2f} {b[f'rec_{c}']:.2f} {b[f'F1_{c}']:.2f}")
print("\n== per fold: sample EM test (A / B), val sample EM at 0.5, best epoch / epochs run / minutes")
for k in range(5):
    m = PA["fold"].values == k
    print(f"fold {k}: test {hA[m].mean():.3f} / {hB[m].mean():.3f} | val {infA[k][3]:.3f} / {infB[k][3]:.3f}"
          f" | thr {tA[k]} / {tB[k]} | {infA[k][:3]} / {infB[k][:3]}")
print(f"\n== difference {B.name} - {A.name}, sample EM: {hB.mean() - hA.mean():+.4f}"
      f" (paired bootstrap 95% CI {dlo:+.4f} to {dhi:+.4f})")
print("== exact McNemar (b = only A right, c = only B right)")
for k, (b_, c_, p) in tests.items():
    print(f"  {k:42s} b={b_:3d} c={c_:3d} p={p:.4f}")
if len(sys.argv) > 3:
    out.to_csv(sys.argv[3]); print("saved", sys.argv[3])
