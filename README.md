# semi_supervised_image_clf

**Label-Efficient Image Classification: From Self-Supervised Pretraining to Semi-Supervised Fine-Tuning**

Implements and compares three approaches on STL-10 to show how much SSL pretraining reduces the label requirement.

### Measured results (Kaggle T4, April 2026)

SimCLR ran 20 epochs; FixMatch sweep run separately (SimCLR + FixMatch end-to-end pending).

| Method | Labels / class | Accuracy |
|---|---|---|
| Supervised baseline | 500 | **55.9%** |
| FixMatch (random init) | 100 | **41.6%** |
| FixMatch (random init) | 500 | **61.8%** |
| FixMatch (random init) | 4 / 10 / 25 | ~10% ⚠ |
| SimCLR linear probe | — (20 epochs) | **55.5%** |
| SimCLR + FixMatch | — | *pending* |

> ⚠ FixMatch at ≤25 labels/class scores near-random: the 0.95 pseudo-label confidence
> threshold is never met with a random encoder + few labels. The SimCLR encoder init
> is expected to resolve this.

### Projected (100-epoch SimCLR + combined sweep)

| Method | Labels / class | Projected accuracy |
|---|---|---|
| Supervised baseline | 500 | ~72% |
| FixMatch (random init) | 100 | ~68% |
| FixMatch (random init) | 25 | ~58% |
| **SimCLR + FixMatch** | **100** | **~75%** |
| **SimCLR + FixMatch** | **25** | **~70%** |
| **SimCLR + FixMatch** | **4** | **~60%** |

SimCLR + FixMatch with **25 labels/class** (~70%) is projected to match a fully supervised model at **500 labels/class** (~72%) — a **20× label reduction**.

---

## Methods

**SimCLR** (Chen et al., 2020) — self-supervised pretraining via contrastive learning. Two augmented views of the same image are pulled together in embedding space using NT-Xent loss. The projection head is discarded; only the encoder is kept.

**FixMatch** (Sohn et al., 2020) — semi-supervised learning. A model trained on labelled data generates pseudo-labels from weak augmentations of unlabelled images. High-confidence pseudo-labels are used as supervision for strongly augmented versions of the same images.

**SimCLR + FixMatch** — SimCLR pretraining gives the encoder a strong feature representation before FixMatch begins. Better initial features → more accurate pseudo-labels from the start → faster convergence and higher final accuracy, especially at very low label counts.

---

## Quickstart

```bash
# Install dependencies
make sync

# Fast local sanity-check (no GPU, ~120 MB download, 2 epochs)
make smoke-data
make smoke

# Run tests
make test

# Lint
make lint
```

### Full training (Kaggle GPU recommended)

```bash
# 1. Download data
make data             # train + test (~310 MB on disk)
make data-unlabeled   # unlabeled split (~2.5 GB on disk, needed for SimCLR)

# 2. SimCLR pretraining (~90 min on T4)
make pretrain

# 3. Label-fraction sweep (~90 min on T4)
make sweep

# 4. Plot results
make plot

# 5. Export best model
make export
```

Or run the Kaggle notebooks directly:
- [`kaggle/simclr_pretrain.ipynb`](kaggle/simclr_pretrain.ipynb) — SimCLR pretraining (template)
- [`kaggle/fixmatch_sweep.ipynb`](kaggle/fixmatch_sweep.ipynb) — FixMatch label sweep + plots (template)
- [`kaggle/simclr-pretrain.ipynb`](kaggle/simclr-pretrain.ipynb) — trained version with outputs
- [`kaggle/fixmatch-sweep.ipynb`](kaggle/fixmatch-sweep.ipynb) — trained version with outputs

Both notebooks have a `SMOKE_TEST = False` flag at the top for low-disk / quick runs.

---

## Project Structure

```
src/semi_supervised_image_clf/
├── config.py         # Pydantic v2 config schemas + YAML loaders
├── dataset.py        # STL-10 loading, label fraction sampling
├── augmentations.py  # SimCLR, weak, and strong (RandAugment) pipelines
├── model.py          # ResNet-18 backbone, SimCLR head, classifier, EMA
├── simclr.py         # NT-Xent loss, SimCLRDataset, train_simclr
├── fixmatch.py       # PseudoLabelFilter, FixMatchUnlabelledDataset, train_fixmatch
├── supervised.py     # train_supervised
├── evaluate.py       # EvalResult, evaluate, plot_tsne_embeddings
├── export.py         # ONNX export + onnxruntime validation
└── plot.py           # plot_label_efficiency_curve, plot_training_curves
```

---

## Data

**STL-10** — designed for semi-supervised and self-supervised research.

| Split | Size | Use |
|---|---|---|
| train | 5,000 (500/class × 10 classes) | Supervised loss / label sampling |
| unlabelled | 100,000 | SimCLR pretraining / FixMatch consistency loss |
| test | 8,000 | Evaluation |

Download tiers:

```bash
make smoke-data       # train only (~120 MB on disk)
make data             # train + test (~310 MB on disk)
make data-unlabeled   # unlabeled (~2.5 GB on disk)
```

The archive is streamed from the network — the 2.6 GB tar.gz is never written to disk. Only the requested binary files are extracted.

---

## Architecture

- **Backbone**: ResNet-18, random init, 64×64 input, 512-d features
- **SimCLR head**: `Linear(512→512) → BN → ReLU → Linear(512→128)`
- **Classifier head**: `Linear(512→10)` (replaces SimCLR head after pretraining)
- **EMA**: exponential moving average of model weights for stable pseudo-labels

---

## Development

```
make sync         # uv sync --dev
make test         # pytest with coverage (81%, 58 tests)
make lint         # ruff format → ruff check --fix → ty check
make smoke-data   # download train split only (~120 MB)
make smoke        # 2-epoch end-to-end run; uses synthetic data for missing splits
make clean        # remove data/stl10_binary/, checkpoints, plots, mlruns
```

CI runs `make sync && make lint && make test` on every push via GitHub Actions.

`make smoke` works after `make smoke-data` alone — the unlabeled and test splits are replaced with synthetic PIL images so no 2.5 GB download is required on Codespaces.

---

## License

Apache 2.0
