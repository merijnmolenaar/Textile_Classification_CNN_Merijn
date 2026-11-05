"""Labels, image matching and the two input pipelines.

Every dataset returns views of shape (V, 3, S, S):
  resize mode  V = 1   the whole 1920x1080 frame squashed to S x S (Elsherif, ~32 x 18 um/px)
  crop mode    V = 1   at training: one random native-resolution S x S crop (7.58 um/px kept)
               V = c*r at evaluation: an evenly spaced grid of S x S tiles covering the frame
The engine averages the V view probabilities into one image probability, and the
image probabilities of a sample into one sample probability.
"""
import os
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- labels and images

# Words in the composition text -> class. Unknown words raise an error, so a new label
# file with a new spelling fails loudly instead of silently dropping a fibre.
FIBRE_ALIASES = {
    "cotton": "Cotton", "wool": "Wool", "hemp": "Hemp", "linen": "Linen",
    "elastane": "Elastane", "elastine": "Elastane", "ea": "Elastane",   # EA = ISO 2076 code
    "polyester": "Polyester", "viscose": "Viscose", "silk": "Silk", "tencel": "Tencel",
    "pla": "PLA", "polylactide": "PLA", "polyamide": "Polyamide", "poliamide": "Polyamide",
    "alpaca": "Alpaca", "peatfibre": "Peat fibre",
    # SE (Seide) = silk and HE = hemp: the only reading that reproduces the class counts
    # in Elsherif's report (Fig. 9) exactly. Used by s197, s200, s201, s203 ("60cotton 33SE 7HE").
    "se": "Silk", "he": "Hemp",
    "natural": None,   # "100hemp natural"
}


def parse_composition(text, classes):
    """'60cotton 33SE 7HE' -> {'Cotton', 'Silk', 'Hemp'}. Percentages are ignored."""
    out = set()
    for w in re.findall(r"[a-z]+", str(text).lower()):
        if w not in FIBRE_ALIASES:
            raise ValueError(f"unknown fibre word '{w}' in composition '{text}'; add it to FIBRE_ALIASES")
        if FIBRE_ALIASES[w]:
            out.add(FIBRE_ALIASES[w])
    unknown = out - set(classes)
    if unknown:
        raise ValueError(f"composition '{text}' gives classes {unknown} not in data.classes")
    return out


def load_labels(xlsx, classes, overrides=None):
    """Label table with columns key + one 0/1 column per class. Accepts two formats:
    - composition text: columns Filename, composition (labels20260217_sort3d_hsi_measurements.xlsx)
    - matrix: columns Name + one 0/1 column per class (Elsherif's LabelsMatrix.xlsx)
    Rows without composition are skipped. `overrides` = {sample id: [classes]} replaces parsed labels."""
    df = pd.read_excel(xlsx, sheet_name=0)
    if "composition" in df.columns:
        df = df[df["composition"].notna()].copy()
        df["key"] = df["Filename"].astype(str).str.lower().str.strip()
        sets = [parse_composition(t, classes) for t in df["composition"]]
        for c in classes:
            df[c] = [int(c in s) for s in sets]
    elif "Name" in df.columns:
        missing = [c for c in classes if c not in df.columns]
        if missing:
            raise ValueError(f"{xlsx} lacks class columns {missing}")
        df["key"] = df["Name"].astype(str).str.lower().str.strip()
        for c in classes:
            df[c] = df[c].fillna(0).astype(int)
    else:
        raise ValueError(f"{xlsx}: expected a 'composition' or a 'Name' column; found {list(df.columns)}")
    for sid, labs in (overrides or {}).items():
        i = df.index[df["key"] == sid.lower()]
        if len(i):
            for c in classes:
                df.loc[i, c] = int(c in labs)
    return df[["key", *classes]].reset_index(drop=True)


def index_images(image_dirs):
    """sample id -> sorted list of image paths, over one or more folders."""
    by_name = {}
    for d in image_dirs:
        for p in sorted(Path(d).glob("*.jpg")):
            by_name.setdefault(p.name, str(p))      # first folder wins on duplicates
    s2i = defaultdict(list)
    for name, path in sorted(by_name.items()):
        sid = re.sub(r"_\d+$", "", Path(name).stem).lower()   # S001_2 -> s001
        s2i[sid].append(path)
    return s2i


def build_index(cfg):
    """Returns (samples, classes, skipped): samples = ordered list of (sid, label vector, [paths]);
    skipped = {excluded class: [sample ids left out because they contain it]}.

    Order follows the label file, as in Elsherif's notebook, so his split can be rebuilt.
    Classes with no positive sample are dropped, as in his notebook.
    `data.exclude_classes`: samples containing any of these classes are left out, and the
    classes are removed from the output. The label file is parsed with the full class list
    first, so a sample like "50PEATfibre 50cotton" is skipped, not relabelled as pure cotton.
    """
    dc = cfg["data"]
    df = load_labels(dc["labels_xlsx"], dc["classes"], dc.get("label_overrides"))
    s2i = index_images(dc["image_dirs"])
    n_img = sum(len(v) for v in s2i.values())
    excl = {s.lower() for s in dc.get("exclude") or []}
    keys = [k for k in df["key"] if k in s2i and k not in excl]
    unlabelled = sorted(set(s2i) - set(df["key"]))
    lab = df.set_index("key")[dc["classes"]]

    excl_cls = list(dc.get("exclude_classes") or [])
    unknown = [c for c in excl_cls if c not in dc["classes"]]
    if unknown:
        raise ValueError(f"data.exclude_classes {unknown} not in data.classes")
    skipped = {c: [k for k in keys if lab.at[k, c] == 1] for c in excl_cls}
    gone = {k for ids in skipped.values() for k in ids}
    keys = [k for k in keys if k not in gone]
    use = [c for c in dc["classes"] if c not in excl_cls]

    Y = np.stack([lab.loc[k, use].to_numpy(dtype=np.float32) for k in keys])
    keep = Y.sum(0) > 0
    classes = [c for c, k in zip(use, keep) if k]
    dropped = [c for c, k in zip(use, keep) if not k]
    Y = Y[:, keep]
    samples = [(k, Y[i], s2i[k]) for i, k in enumerate(keys)]
    print(f"{n_img} images | {len(s2i)} sample ids | {len(samples)} matched to labels"
          + (f" | excluded: {sorted(excl)}" if excl else "")
          + (f" | excluded classes {excl_cls}: {len(gone)} samples skipped {sorted(gone)}" if excl_cls else "")
          + (f" | unlabelled: {unlabelled}" if unlabelled else "")
          + (f" | dropped empty classes: {dropped}" if dropped else ""))
    return samples, classes, skipped


def records(samples, ids):
    ids = set(ids)
    return [(p, y, sid) for sid, y, paths in samples if sid in ids for p in paths]


# ---------------------------------------------------------------- transforms

class AddGaussianNoise:
    def __init__(self, std): self.std = std
    def __call__(self, t): return t + torch.randn_like(t) * self.std


def _norm():
    return [T.ToTensor(), T.Normalize(MEAN, STD)]


def resize_transforms(S, augment, a):
    eval_tf = T.Compose([T.Resize((S, S)), *_norm()])
    if not augment:
        return eval_tf, eval_tf
    train_tf = T.Compose([   # Elsherif's augmentation, unchanged
        T.RandomResizedCrop(S, scale=tuple(a["crop_scale"]), ratio=tuple(a["crop_ratio"])),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(a["rotation_deg"]),
        T.ColorJitter(contrast=a["contrast"], saturation=a["saturation"]),
        *_norm(), AddGaussianNoise(a["noise_std"])])
    return train_tf, eval_tf


class RandomRot90:
    """Rotate by 0, 90, 180 or 270 degrees. On a square crop this leaves no empty corners,
    unlike a free rotation (which fills the corners with black)."""
    def __call__(self, im):
        k = int(torch.randint(4, (1,)))
        return im.rotate(90 * k) if k else im


def crop_train_transform(S, augment, a):
    # No rescaling in crop mode: a random-resized crop would change um/px, which is the
    # variable under study. Only flips, rotation, colour jitter and noise.
    # rotation_deg 0 switches free rotation off; noise_std 0 switches noise off.
    if not augment:
        return T.Compose([T.RandomCrop(S), *_norm()])
    geo = [T.RandomCrop(S), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    if a.get("rot90"):
        geo.append(RandomRot90())
    if a.get("rotation_deg"):
        geo.append(T.RandomRotation(a["rotation_deg"]))
    colour = T.ColorJitter(brightness=a.get("brightness", 0), contrast=a.get("contrast", 0),
                           saturation=a.get("saturation", 0), hue=a.get("hue", 0))
    noise = [AddGaussianNoise(a["noise_std"])] if a.get("noise_std") else []
    return T.Compose([*geo, colour, *_norm(), *noise])


def tile_boxes(W, H, S, cols, rows):
    """Evenly spaced S x S boxes covering a W x H frame (overlap when needed)."""
    xs = np.linspace(0, W - S, cols).round().astype(int) if cols > 1 else [(W - S) // 2]
    ys = np.linspace(0, H - S, rows).round().astype(int) if rows > 1 else [(H - S) // 2]
    return [(x, y, x + S, y + S) for y in ys for x in xs]


# ---------------------------------------------------------------- datasets

def _open(p):
    with Image.open(p) as im:
        return im.convert("RGB")


class ImageDataset(Dataset):
    """Training or evaluation data for either mode. Item: (views[V,3,S,S], label, sid, path)."""

    def __init__(self, recs, cfg, train):
        dc = cfg["data"]
        self.recs, self.train, self.mode, self.S = recs, train, dc["mode"], dc["img_size"]
        self.reps = dc["train_crops_per_image"] if (train and self.mode == "crop") else 1
        if self.mode == "resize":
            tr, ev = resize_transforms(self.S, dc["augment"], dc["aug"])
            self.tf = tr if train else ev
        elif self.mode == "crop":
            self.tf = crop_train_transform(self.S, dc["augment"], dc["aug"]) if train else T.Compose(_norm())
            self.grid = tuple(dc["eval_tiles"])
        else:
            raise ValueError(f"unknown data.mode {self.mode}")

    @property
    def views(self):
        return self.grid[0] * self.grid[1] if (self.mode == "crop" and not self.train) else 1

    def __len__(self):
        return len(self.recs) * self.reps

    def __getitem__(self, i):
        p, y, sid = self.recs[i % len(self.recs)]
        im = _open(p)
        if self.mode == "crop" and not self.train:
            boxes = tile_boxes(*im.size, self.S, *self.grid)
            x = torch.stack([self.tf(im.crop(b)) for b in boxes])
        else:
            x = self.tf(im).unsqueeze(0)
        return x, torch.from_numpy(y), sid, p


# ---------------------------------------------------------------- Colab: copy data to local disk

def cache_locally(cfg):
    """Copy image folders and label file to data.cache_dir once; rewrite cfg paths to the copy.
    Reading 2208 JPEGs from Google Drive every epoch is slow; local disk is not."""
    dc = cfg["data"]
    dst_root = dc.get("cache_dir")
    if not dst_root:
        return cfg
    jobs, new_dirs = [], []
    for i, src in enumerate(dc["image_dirs"]):
        dst = os.path.join(dst_root, f"images_{i}")
        os.makedirs(dst, exist_ok=True)
        new_dirs.append(dst)
        for f in os.listdir(src):
            if f.lower().endswith(".jpg") and not os.path.exists(os.path.join(dst, f)):
                jobs.append((os.path.join(src, f), os.path.join(dst, f)))
    lab_dst = os.path.join(dst_root, os.path.basename(dc["labels_xlsx"]))
    shutil.copy2(dc["labels_xlsx"], lab_dst)
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda a: shutil.copy2(*a), jobs))
    print(f"cached {len(jobs)} new images to {dst_root} in {time.time() - t0:.0f} s")
    dc["image_dirs"], dc["labels_xlsx"] = new_dirs, lab_dst
    return cfg
