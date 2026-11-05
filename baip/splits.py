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


def subsample_train(tr, samples, frac, seed):
    """Keep a fraction of the training samples, for learning curves. Validation and test are
    untouched, so every fraction is scored on exactly the same samples as the full run.

    Stratified by exact label combination: each combination keeps round(frac * n) samples
    (rounding half up), so rare combinations can drop out at small fractions, as they would
    in a smaller dataset. Nested: with the same seed, the 25% set is contained in the 50% set,
    and so on, because the order within each combination is fixed by the seed only.
    frac = 1.0 returns the training set unchanged (the v1 runs).
    """
    if frac >= 1.0:
        return tr
    if not 0.0 < frac < 1.0:
        raise ValueError(f"split.train_frac must be in (0, 1], got {frac}")
    rng = np.random.default_rng(seed)
    combos = np.array(["".join(str(int(v)) for v in samples[i][1]) for i in tr])
    keep = []
    for c in sorted(set(combos)):
        idx = np.asarray(tr)[combos == c]
        idx = idx[rng.permutation(len(idx))]
        keep.extend(idx[: int(np.floor(frac * len(idx) + 0.5))].tolist())
    return np.array(sorted(keep))


def split_rarest_fibre(samples, classes, fracs=(0.70, 0.15, 0.15), seed=42, min_blend=5):
    """Train/val/test split by sample, stratified on the rarest fibre of each sample.

    1. Each sample gets the group of its rarest fibre (by count over all samples), so
       "cotton + elastane" goes with elastane and "wool + PLA" with PLA.
    2. Within that group, every blend (exact label combination) with >= `min_blend` samples
       is split on its own, so it appears in train, val and test. Smaller blends are pooled
       within the group.
    3. Each unit gets round(frac * n) samples for val and test (at least 1 each when n >= 3),
       the rest go to train. Samples are interleaved over blends and dealt by remaining need,
       so a pooled unit spreads its blends over the partitions.
    Returns {"train": [...], "val": [...], "test": [...]} with sample ids. Deterministic for a seed.
    """
    ids = [s[0] for s in samples]
    Y = np.stack([s[1] for s in samples]).astype(int)
    count = Y.sum(0)
    combo = [" + ".join(c for c, v in zip(classes, y) if v) for y in Y]
    rarest = [min((i for i in range(len(classes)) if y[i]), key=lambda i: count[i]) for y in Y]
    n_combo = Counter(combo)
    unit = [f"{classes[r]} | {c if n_combo[c] >= min_blend else 'small blends'}" for r, c in zip(rarest, combo)]

    rng = np.random.default_rng(seed)
    fr = np.asarray(fracs, dtype=float)
    parts = ("train", "val", "test")
    out = {p: [] for p in parts}
    ids_a, combo_a, unit_a = np.array(ids), np.array(combo), np.array(unit)
    for u in sorted(set(unit)):
        members = ids_a[unit_a == u]
        mcombo = combo_a[unit_a == u]
        queues = [list(rng.permutation(members[mcombo == c])) for c in sorted(set(mcombo))]
        order = []
        while any(queues):
            for q in queues:
                if q:
                    order.append(q.pop())
        n = len(order)
        target = np.floor(fr * n + 0.5).astype(int)
        if n >= 3:
            target[1:] = np.maximum(target[1:], 1)
        target[0] = n - target[1:].sum()
        got = np.zeros(3, int)
        for sid in order:
            need = (target - got) / np.maximum(target, 1)
            need[got >= target] = -9
            k = int(np.argmax(need))
            got[k] += 1
            out[parts[k]].append(str(sid))
    order_of = {s: i for i, s in enumerate(ids)}
    return {p: sorted(v, key=order_of.get) for p, v in out.items()}


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
        tr = subsample_train(tr, samples, sc.get("train_frac", 1.0), sc.get("subsample_seed", 0))
        return {"scheme": scheme, "fold": sc["fold"], "n_folds": k,
                "train_frac": sc.get("train_frac", 1.0), "subsample_seed": sc.get("subsample_seed", 0),
                "train": list(ids[tr]), "val": list(ids[va]), "test": list(ids[te])}

    raise ValueError(f"unknown split.scheme {scheme}")


def get_split(samples, sc):
    if sc.get("split_file"):
        with open(sc["split_file"]) as fh:
            split = json.load(fh)
        # a saved split must cover exactly the samples of this run (same label file, same exclusions)
        have = {s[0] for s in samples}
        listed = set(split["train"]) | set(split["val"]) | set(split["test"])
        if listed != have:
            raise ValueError(f"split_file {sc['split_file']} does not match the samples of this run: "
                             f"{len(listed - have)} listed but not loaded (e.g. {sorted(listed - have)[:5]}), "
                             f"{len(have - listed)} loaded but not listed (e.g. {sorted(have - listed)[:5]})")
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
