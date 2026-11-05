"""Run configuration.

A run is built from three layers, later ones win:
  1. DEFAULTS below (Elsherif's notebook 02 settings)
  2. a paths file (where the data lives on this machine), e.g. configs/paths_local.yaml
  3. an experiment file, e.g. configs/crop448_holdout.yaml
  4. command-line overrides: --set train.lr=1e-4 --set data.mode=crop
"""
import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "name": "unnamed",
    "seed": 42,
    "data": {
        "image_dirs": [],           # one or more folders with S###_k.jpg
        "labels_xlsx": None,        # composition-text file or Elsherif-style label matrix
        # Samples whose composition text is ambiguous. These follow Elsherif's labelling, which
        # was reconstructed from the class counts in his report (Fig. 9); see notes/decisions.md.
        "label_overrides": {
            "s731": ["Wool"], "s734": ["Wool"], "s735": ["Wool"], "s736": ["Wool"],   # "100cotton/100wool", double sided
            "s737": ["Wool"], "s738": ["Wool"],                                       # same, no images
            "s315": ["Wool"],                                                         # "?hemp ?wool"
        },
        "exclude": [],              # sample ids left out entirely
        "classes": ["Cotton", "Wool", "Hemp", "Linen", "Elastane", "Polyester",
                    "Viscose", "Silk", "Tencel", "PLA", "Polyamide", "Alpaca", "Peat fibre"],
        # Classes left out of the model. Every sample that contains one of them is skipped
        # (the label file itself is not changed), and the class is removed from the output.
        # Skipped samples are listed in <run>/excluded_samples.json.
        "exclude_classes": [],
        "um_per_px": 7.58,          # logged with every run; the RQ is about this number
        "mode": "resize",           # "resize": whole frame squashed to img_size (Elsherif)
                                    # "crop":   native-resolution img_size crops, no resizing
        "img_size": 448,
        "eval_tiles": [5, 3],       # crop mode: tile grid (cols, rows) at evaluation, evenly spaced
        "train_crops_per_image": 1, # crop mode: random crops per image per epoch
        "augment": False,
        "aug": {"crop_scale": [0.6, 1.0], "crop_ratio": [0.9, 1.1], "rotation_deg": 15,
                "contrast": 0.2, "saturation": 0.2, "noise_std": 0.01},
        "num_workers": 4,
        "cache_dir": None,          # Colab: copy images to local disk first (much faster than Drive)
    },
    "split": {
        "scheme": "holdout",        # "elsherif": 80/20 train/val, no test (reproduction only)
                                    # "holdout":  train/val/test by sample, stratified
                                    # "cv":       k folds by sample; test = fold `fold`, val carved from the rest
        "val_frac": 0.15,
        "test_frac": 0.15,
        "n_folds": 5,
        "fold": 0,
        "split_seed": 42,           # kept separate from `seed` so all experiments share one split
        "split_file": None,         # optional: reuse a saved split.json
    },
    "model": {"backbone": "densenet121", "pretrained": True},
    "train": {
        "lr": 1e-3, "weight_decay": 0.01, "eta_min": 1e-6,
        "epochs": 40, "patience": 10, "batch_size": 16,
        "select_on": "val_sample_acc",  # "val_image_acc" reproduces Elsherif
        "threshold": 0.5,
        "tune_threshold": True,         # pick a global threshold on validation, apply to test
        "pos_weight": False,            # class-weighted BCE (later experiment)
        "amp": True,                    # CUDA only: bf16 (A100/L4) or fp16 + loss scaling (T4); ignored on MPS/CPU
        "eval_batch_tiles": 32,         # tiles per forward pass at evaluation (memory only)
        "max_batches": None,            # smoke test / timing: stop each epoch early
    },
    "out_dir": "results",
    "run_dir": None,                # set per fold by scripts/run_cv.py; normally leave empty
    "ckpt_dir": "checkpoints",
}


def _merge(base, new):
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _parse_value(s):
    return yaml.safe_load(s)


def load_config(files=(), overrides=()):
    cfg = copy.deepcopy(DEFAULTS)
    for f in files:
        if f:
            with open(f) as fh:
                _merge(cfg, yaml.safe_load(fh))
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = _parse_value(val)
    return cfg


def dump_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))
