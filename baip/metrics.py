"""Image- and sample-level metrics. The main metric is sample-level exact-match accuracy."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_recall_fscore_support

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def exact_match(L, P):
    return float(np.all(L == P, axis=1).mean()) if len(L) else float("nan")


def to_samples(probs, labels, sids):
    """Average image probabilities per sample. Returns (sample ids, probs, labels)."""
    by, lab = defaultdict(list), {}
    for p, l, s in zip(probs, labels, sids):
        by[s].append(p)
        lab[s] = l
    ids = sorted(by)
    return ids, np.array([np.mean(by[s], 0) for s in ids]), np.array([lab[s] for s in ids]).astype(int)


def tune_threshold(probs, labels):
    """Global threshold that maximises exact-match. Call on validation data only."""
    accs = [exact_match(labels, (probs > t).astype(int)) for t in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.argmax(accs))])


def bootstrap_ci(L, P, n=2000, seed=0):
    """95% CI of exact-match accuracy, resampling rows (pass sample-level arrays)."""
    if not len(L):
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    hit = np.all(L == P, axis=1).astype(float)
    accs = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(n)]
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def report(L, P, classes):
    prec, rec, f1, sup = precision_recall_fscore_support(L, P, average=None, zero_division=0)
    return {
        "n": int(len(L)),
        "exact_match": exact_match(L, P),
        "exact_match_ci95": bootstrap_ci(L, P),
        "macro_f1": float(f1_score(L, P, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(L, P, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(L, P, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(L, P)),
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]),
                          "f1": float(f1[i]), "support": int(sup[i])}
                      for i, c in enumerate(classes)},
    }


def evaluate(pred, classes, threshold):
    """pred = dict(probs, labels, sids) at image level. Returns image- and sample-level reports."""
    Li = pred["labels"].astype(int)
    img = report(Li, (pred["probs"] > threshold).astype(int), classes)
    ids, sp, sl = to_samples(pred["probs"], Li, pred["sids"])
    smp = report(sl, (sp > threshold).astype(int), classes)
    return {"threshold": threshold, "image": img, "sample": smp}
