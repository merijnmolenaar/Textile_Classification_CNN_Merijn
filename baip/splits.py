"""Sample-level splits. All images of one sample always land in the same partition."""
import json
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def strata(samples, min_count):
    """Stratum per sample = its label combination; combinations seen fewer than
    `min_count` times are pooled into one 'rare' stratum so stratification is possible."""
    combos = ["".join(str(int(v)) for v in y) for _, y, _ in samples]
    counts = Counter(combos)
    out = np.array([c if counts[c] >= min_count else "rare" for c in combos])
    if (out == "rare").sum() == 1:   # a stratum of one cannot be split; give it to the largest
        out[out == "rare"] = Counter(out.tolist()).most_common(1)[0][0]
    return out


def make_split(samples, sc):
    ids = np.array([s[0] for s in samples])
    seed = sc["split_seed"]
    scheme = sc["scheme"]

    if scheme == "elsherif":
        # Exactly notebook 02: ids in label-file order, 80/20, no stratification, no test set.
        tr, va = train_test_split(list(ids), test_size=0.2, random_state=seed, shuffle=True)
        return {"scheme": scheme, "train": tr, "val": va, "test": []}

    if scheme == "holdout":
        st = strata(samples, min_count=6)
        rest, te = train_test_split(np.arange(len(ids)), test_size=sc["test_frac"],
                                    random_state=seed, stratify=st)
        st_rest = strata([samples[i] for i in rest], min_count=4)
        tr, va = train_test_split(rest, test_size=sc["val_frac"] / (1 - sc["test_frac"]),
                                  random_state=seed, stratify=st_rest)
        return {"scheme": scheme, "train": list(ids[tr]), "val": list(ids[va]), "test": list(ids[te])}

    if scheme == "cv":
        k = sc["n_folds"]
        st = strata(samples, min_count=k)
        folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(ids, st))
        rest, te = folds[sc["fold"]]
        st_rest = strata([samples[i] for i in rest], min_count=4)
        tr, va = train_test_split(rest, test_size=sc["val_frac"] / (1 - 1 / k),
                                  random_state=seed, stratify=st_rest)
        return {"scheme": scheme, "fold": sc["fold"], "n_folds": k,
                "train": list(ids[tr]), "val": list(ids[va]), "test": list(ids[te])}

    raise ValueError(f"unknown split.scheme {scheme}")


def get_split(samples, sc):
    if sc.get("split_file"):
        with open(sc["split_file"]) as fh:
            split = json.load(fh)
    else:
        split = make_split(samples, sc)
    parts = [set(split[k]) for k in ("train", "val", "test")]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]), "split overlaps"
    return split


def describe(split, samples, classes):
    """Positive samples per class per partition, to spot rare fibres missing from val/test."""
    y = {sid: lab for sid, lab, _ in samples}
    rows = {}
    for part in ("train", "val", "test"):
        ids = split[part]
        rows[part] = (np.sum([y[s] for s in ids], 0).astype(int).tolist() if ids else [0] * len(classes))
    w = max(len(c) for c in classes)
    lines = [f"{'class':<{w}}  train    val   test"]
    for i, c in enumerate(classes):
        lines.append(f"{c:<{w}}  {rows['train'][i]:>5}  {rows['val'][i]:>5}  {rows['test'][i]:>5}")
    lines.append(f"{'samples':<{w}}  {len(split['train']):>5}  {len(split['val']):>5}  {len(split['test']):>5}")
    return "\n".join(lines), rows
