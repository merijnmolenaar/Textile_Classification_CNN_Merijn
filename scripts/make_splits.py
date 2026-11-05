"""Make a fixed train/val/test split by sample and save it, so experiments read it instead of re-splitting.

Method `rarest_fibre` (see baip.splits.split_rarest_fibre): stratified on the rarest fibre of each
sample, and within that on the blend (label combination) for blends with >= 5 samples.

Checks before saving: no sample in two partitions, every loaded sample in exactly one partition
(so all 3 images of a sample stay together), every fibre present in val and test, and the same seed
gives the same split. Prints count tables per fibre, per blend and per number of fibres.

Example (the split of experiment exp/holdout-aug-rare):
  python scripts/make_splits.py --paths configs/paths_local.yaml \
      --exclude-classes "Peat fibre,Polyamide,Alpaca" --out splits/holdout_70_15_15_v1.json
Use it in an experiment config with:  split: {split_file: splits/holdout_70_15_15_v1.json}
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baip import data as D                                   # noqa: E402
from baip.config import load_config                          # noqa: E402
from baip.splits import split_rarest_fibre                   # noqa: E402

PARTS = ("train", "val", "test")


def tables(split, samples, classes):
    y = {sid: lab.astype(int) for sid, lab, _ in samples}
    part = {sid: p for p in PARTS for sid in split[p]}
    df = pd.DataFrame([y[s] for s in part], index=list(part), columns=classes)
    df["part"] = [part[s] for s in df.index]
    df["blend"] = [" + ".join(c for c in classes if df.at[s, c]) for s in df.index]
    df["n_fibres"] = df[classes].sum(axis=1)

    fib = pd.DataFrame({p: df.loc[df.part == p, classes].sum() for p in PARTS})
    fib["total"] = fib.sum(axis=1)
    for p in PARTS:
        fib[f"{p}_%"] = (100 * fib[p] / fib["total"]).round(1)
    blend = pd.crosstab(df.blend, df.part).reindex(columns=PARTS, fill_value=0)
    blend["total"] = blend.sum(axis=1)
    blend = blend.sort_values("total", ascending=False)
    nf = pd.crosstab(df.n_fibres, df.part).reindex(columns=PARTS, fill_value=0)
    nf["total"] = nf.sum(axis=1)
    return fib, blend, nf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="paths file (image folders, label file)")
    ap.add_argument("--exclude-classes", default="", help="comma-separated, as data.exclude_classes")
    ap.add_argument("--fracs", default="0.70,0.15,0.15", help="train,val,test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-blend", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    excl = [c.strip() for c in a.exclude_classes.split(",") if c.strip()]
    fracs = tuple(float(x) for x in a.fracs.split(","))
    assert len(fracs) == 3 and abs(sum(fracs) - 1) < 1e-9, "--fracs must be three numbers summing to 1"
    cfg = load_config([a.paths], [f"data.exclude_classes={json.dumps(excl)}"])
    samples, classes, skipped = D.build_index(cfg)

    split = split_rarest_fibre(samples, classes, fracs, a.seed, a.min_blend)

    # checks
    sets = [set(split[p]) for p in PARTS]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "a sample is in two partitions"
    assert set().union(*sets) == {s[0] for s in samples}, "not every sample is in a partition"
    assert sum(len(x) for x in sets) == len(samples)
    assert split == split_rarest_fibre(samples, classes, fracs, a.seed, a.min_blend), "split not deterministic"
    fib, blend, nf = tables(split, samples, classes)
    empty = [c for c in classes if fib.at[c, "val"] == 0 or fib.at[c, "test"] == 0]
    assert not empty, f"fibres missing from val or test: {empty}"

    label_file = Path(cfg["data"]["labels_xlsx"])
    out = {
        "scheme": "holdout_rarest_fibre",
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "method": "stratified on the rarest fibre per sample; blends with >= min_blend samples split on their own",
        "fracs": dict(zip(PARTS, fracs)), "seed": a.seed, "min_blend": a.min_blend,
        "exclude_classes": excl, "excluded_samples": skipped,
        "label_file": label_file.name,
        "label_file_sha256": hashlib.sha256(label_file.read_bytes()).hexdigest(),
        "classes": classes,
        "n_samples": {p: len(split[p]) for p in PARTS},
        "class_counts": {p: fib[p].astype(int).to_dict() for p in PARTS},
        "n_fibres_counts": {p: {int(k): int(v) for k, v in nf[p].items()} for p in PARTS},
        **split,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))

    pd.set_option("display.width", 200)
    print(f"\nsamples: {out['n_samples']} (images x3)")
    print("\nper fibre:\n" + fib.to_string())
    print("\nper number of fibres in the sample:\n" + nf.to_string())
    print("\nper blend:\n" + blend.to_string())
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
