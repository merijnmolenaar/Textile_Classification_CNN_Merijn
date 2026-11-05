# BaIP 2026-27: fibre identification from RGB microscope images

Merijn Molenaar, University of Groningen, S3ORTED WP1. Builds on O. Elsherif's notebooks (below, unchanged).
The notebook pipeline (`02_Microscope_...ipynb`) is ported to plain Python in `baip/`, with three changes:

1. **Native-resolution crops** (`data.mode: crop`). Elsherif squashed each 1920 x 1080 frame to 448 x 448
   (~32 x 18 um/px; a 12 um fibre becomes < 1 px). Crop mode keeps 7.58 um/px: random 448 x 448 crops for
   training, a 5 x 3 grid of tiles covering the whole frame at evaluation, probabilities averaged per image
   and then per sample.
2. **An untouched test set.** Splits are by sample (all images of a sample in one partition), stratified on
   the label combination: 5-fold cross-validation (or a 70/15/15 hold-out). Validation picks the epoch and the threshold; the
   test set is scored once.
3. **Logging.** Every run writes config, split, per-epoch history, metrics (image and sample level, per fibre,
   bootstrap 95% CI) and predictions to `results/<run>/`, plus one line in `results/runs.csv` with the git commit.

## Layout

| Path | What |
|---|---|
| `baip/config.py` | defaults (Elsherif's settings) and config merging |
| `baip/data.py` | label file, image matching, resize and crop pipelines, Colab local-disk cache |
| `baip/splits.py` | sample-level splits: `elsherif` (80/20, reproduction), `holdout`, `cv` |
| `baip/model.py` | DenseNet-121 + head, LAMB optimiser |
| `baip/metrics.py` | exact-match, F1, per-fibre P/R, threshold tuning, bootstrap CI |
| `baip/engine.py` | training loop, evaluation, run logging |
| `baip/cv.py` | pools the test predictions of all folds into one result |
| `scripts/train.py` | one run |
| `scripts/run_cv.py` | k-fold cross-validation with resume, then pooling |
| `scripts/run_queue.py` | runs a list of CV experiments one after another (overnight), one log per experiment |
| `scripts/make_splits.py` | makes a fixed train/val/test split by sample (stratified on the rarest fibre, then on the blend), checks it and saves it to `splits/` |
| `splits/` | fixed splits, committed; experiments read them with `split.split_file` and never re-split |
| `scripts/analysis/` | `compare_runs.py` (paired comparison + McNemar), `learning_curve.py` (power-law fit), `blend_breakdown.py` (single fibre vs blends), `one_vs_three_photos.py`, `plot_history.py` |
| `configs/` | `paths_*.yaml` (where data lives) and one yaml per experiment; `configs/learning_curve/` for the learning curve |
| `colab/run_experiment.ipynb` | Colab runner (GPU) |
| `colab/run_learning_curve.ipynb` | Colab overnight runner for the learning curve |

## Run

```bash
# 5-fold cross-validation (the benchmark). Finished folds are skipped, so rerun the same command to resume.
python scripts/run_cv.py configs/crop448_cv.yaml --paths configs/paths_colab.yaml
python scripts/run_cv.py configs/crop448_cv.yaml --paths configs/paths_colab.yaml --folds 0 1   # part of the folds
python scripts/run_cv.py configs/crop448_cv.yaml --paths configs/paths_colab.yaml --aggregate-only

# single run (reproduction of Elsherif, or a quick check)
python scripts/train.py configs/repro_elsherif.yaml --paths configs/paths_colab.yaml

# Mac: smoke tests only (batch 16 at 448 px does not fit in 16 GB)
python scripts/run_cv.py configs/crop448_cv.yaml --paths configs/paths_local.yaml --folds 0 \
    --set train.epochs=1 --set train.max_batches=2 --set train.batch_size=4 \
    --set train.eval_batch_tiles=8 --set out_dir=results/smoke
```

On Colab: open `colab/run_experiment.ipynb`, choose the experiments, run all cells.

| Experiment | What |
|---|---|
| `repro_elsherif` | Task 1. Elsherif's 80/20 split, no test set. Should land near 74.6% image / 76.2% sample level |
| `resize448_cv` | Task 2. Honest baseline: his resize pipeline, 5-fold CV by sample |
| `crop448_cv` | Task 3. Native-resolution 448 x 448 crops, same folds |
| `crop1080_cv` | Larger context: 1080 x 1080 crops, same folds. Needs a 40 GB+ GPU at batch 16 |

Cross-validation: per fold, test = that fold (20% of samples), validation = 15% of samples from the rest
(picks epoch and threshold), train = the remaining ~65%. All `*_cv` experiments use the same folds
(`split.split_seed: 42`). The pooled result (`<group>/cv_summary.txt`, `cv_metrics.json`,
`cv_sample_predictions.csv`) has one test prediction for every sample; that is the number to report.
Any setting can be overridden with `--set key=value`.

### Learning curve (power-law slope)

`split.train_frac` (default 1.0) keeps only a fraction of each fold's training samples; validation
and test are untouched, so every fraction is scored on the same samples as the full run. The
subsample is stratified by label combination and nested (25% is inside 50% for the same
`split.subsample_seed`). `configs/learning_curve/` has 25/50/75% configs for crop and resize and a
queue file. Run overnight with `colab/run_learning_curve.ipynb`, or:

```bash
python scripts/run_queue.py configs/learning_curve/queue.txt --paths configs/paths_colab.yaml
python scripts/analysis/learning_curve.py /content/drive/MyDrive/BaIP/results   # fit after the queue
```

The 100% points are the existing runs `crop448_cv_v1` and `resize448_cv_v1`.

### Fixed hold-out split with 10 classes (branch `exp/holdout-aug-rare`)

`data.exclude_classes: [Peat fibre, Polyamide, Alpaca]` leaves out the 10 samples containing these
fibres (2, 4 and 4 samples; too few to train or test) without changing the label file: 726 samples,
10 classes. `splits/holdout_70_15_15_v1.json` (504 / 111 / 111 samples) was made once with

```bash
python scripts/make_splits.py --paths configs/paths_local.yaml \
    --exclude-classes "Peat fibre,Polyamide,Alpaca" --out splits/holdout_70_15_15_v1.json
```

A config uses it with `split: {split_file: splits/holdout_70_15_15_v1.json}` plus the same
`data.exclude_classes`; the run stops if the split and the loaded samples do not match.

---

# Original README (O. Elsherif)

Project Notebooks

This repository contains the Jupyter notebooks for the textile classification CNN

Data

The data files used in these notebooks are stored in Google Drive. The notebooks are set up to mount Google Drive and load the files from there.

If you are running these notebooks yourself, you will need to update the import/file-loading section at the top of each notebook to point to wherever the files are stored on your end (e.g. your own Google Drive, a local folder, or another storage location).

Running locally

The notebooks currently include a step that uploads files to Google Drive. If you are running the code locally (not in Google Colab), you can remove or comment out this upload step, since it's only needed for working with files in Drive.

This notebooks contain the numbered sequence of experiments to be ran in order to replicate the results in the report 

Notes

 
File paths will need to be adjusted depending on where you store the data.
No large datasets are included directly in this repository — see the notebooks for where the data is expected to be loaded from.

LINKS TO DATASETS
Microscope data: https://drive.google.com/drive/folders/1NnlttktqZY4hxfOZLg47rAe7cELhqUwC?usp=drive_link
NIST: https://drive.google.com/drive/folders/1uyZlze6uyVF9-bYiCeGIXMApSOjLynU5?usp=sharing
TEXMM: https://drive.google.com/drive/folders/1w0vUkxWzDAhOya6olqum1u6Cr69mkuBU?usp=sharing
