"""crop448_cv_v1: how much accuracy is lost using 1 photo per sample instead of 3?

Reads only saved per-image predictions (already averaged over 15 tiles). No retraining.
Threshold per fold: (a) the run's own validation threshold, (b) re-tuned on validation
using the same number of photos as the variant (fair: selection on validation only).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, str(Path.home() / "Desktop/BaIP/Textile_Classification_CNN_Merijn"))
from baip.metrics import bootstrap_ci, tune_threshold

R = Path.home() / "Desktop/BaIP/results/crop448_cv_v1"
CLASSES = ["Cotton", "Wool", "Hemp", "Linen", "Elastane", "Polyester", "Viscose", "Silk",
           "Tencel", "PLA", "Polyamide", "Alpaca", "Peat fibre"]
PC, YC = [f"p_{c}" for c in CLASSES], [f"y_{c}" for c in CLASSES]
EL = CLASSES.index("Elastane")


def per_sample(df, photos):
    df = df.assign(k=df["image"].str.extract(r"_(\d+)\.jpg$")[0].astype(int))
    if photos != "all":
        df = df[df.k == photos]
    g = df.groupby("sample")
    return g[PC].mean().sort_index(), g[YC].first().sort_index().astype(int)


rows, hits = [], {}
for variant in ["all", 1, 2, 3]:
    for tmode in ["run", "retuned"]:
        allL, allP, sids = [], [], []
        for k in range(5):
            f = R / f"fold{k}"
            thr_run = json.loads((f / "metrics.json").read_text())["val_threshold"]
            Pt, Lt = per_sample(pd.read_csv(f / "preds_test.csv"), variant)
            if tmode == "run":
                thr = thr_run
            else:
                Pv, Lv = per_sample(pd.read_csv(f / "preds_val.csv"), variant)
                thr = tune_threshold(Pv.values, Lv.values)
            allP.append((Pt.values > thr).astype(int)); allL.append(Lt.values); sids += list(Pt.index)
        L, P = np.vstack(allL), np.vstack(allP)
        hit = np.all(L == P, axis=1)
        hits[(variant, tmode)] = pd.Series(hit, index=sids)
        tp = ((L[:, EL] == 1) & (P[:, EL] == 1)).sum()
        rows.append({"photos": "3 (mean)" if variant == "all" else f"only photo {variant}",
                     "threshold": tmode, "n": len(L), "sample_EM": hit.mean(),
                     "ci_lo": bootstrap_ci(L, P)[0], "ci_hi": bootstrap_ci(L, P)[1],
                     "elastane_recall": tp / L[:, EL].sum(), "elastane_n": int(L[:, EL].sum())})

out = pd.DataFrame(rows)
base = hits[("all", "run")]
mc = []
for r in rows:
    v = "all" if r["photos"].startswith("3") else int(r["photos"][-1])
    h = hits[(v, r["threshold"])].reindex(base.index)
    b, c = int((base & ~h).sum()), int((~base & h).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    mc.append({"lost_vs_3": b, "gained_vs_3": c, "mcnemar_exact_p": p})
out = pd.concat([out, pd.DataFrame(mc)], axis=1)
pd.set_option("display.width", 200)
print(out.round(4).to_string(index=False))
dst = Path.home() / "Desktop/BaIP/results/crop448_cv_v1_photos_analysis.csv"
out.to_csv(dst, index=False)
print("saved", dst)
