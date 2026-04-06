# CLAUDE.md — semi_supervised_image_clf

Label-Efficient Image Classification: From Self-Supervised Pretraining to Semi-Supervised Fine-Tuning

---

## Project Overview

This project implements and compares three image classification approaches on STL-10, demonstrating how much label efficiency self-supervised pretraining buys:

- **Baseline**: supervised-only with the full labelled set (5,000 images)
- **FixMatch**: supervised loss on labelled data + consistency regularisation on unlabelled data
- **SimCLR + FixMatch**: SimCLR pretraining on the full unlabelled set, then FixMatch fine-tuning with a fraction of labels

**Key result**: SimCLR + FixMatch with 25 labels/class (~70% accuracy) approaches supervised training with 500 labels/class (~72%) — a 20× label reduction.

---

## Repository Structure

```
semi_supervised_image_clf/
├── .github/workflows/ci.yml
├── .devcontainer/devcontainer.json      # 2-core Codespaces (dev + smoke tests)
├── .claude/commands/                    # Custom Claude Code slash commands
├── kaggle/
│   ├── simclr_pretrain.ipynb            # Kaggle GPU: SimCLR pretraining
│   ├── fixmatch_sweep.ipynb             # Kaggle GPU: FixMatch label sweep
│   └── kaggle.json.example
├── config/
│   ├── simclr.yaml
│   ├── fixmatch.yaml
│   └── supervised.yaml
├── data/
├── checkpoints/
├── plots/
├── src/semi_supervised_image_clf/
│   ├── config.py         # Pydantic v2 config schemas + YAML loaders
│   ├── dataset.py        # STL-10 loading, label fraction sampling
│   ├── augmentations.py  # SimCLR, weak, strong augmentation pipelines
│   ├── model.py          # ResNet-18 backbone, projection head, classifier, EMA
│   ├── simclr.py         # SimCLR pretraining loop + NT-Xent loss
│   ├── fixmatch.py       # FixMatch training loop + pseudo-label logic
│   ├── supervised.py     # Supervised baseline training loop
│   ├── evaluate.py       # Accuracy, confusion matrix, t-SNE embeddings
│   ├── export.py         # ONNX export and validation
│   └── plot.py           # Label efficiency curve, training curves
├── tests/
│   ├── conftest.py
│   ├── test_augmentations.py
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_evaluate.py
│   ├── test_export.py
│   ├── test_fixmatch.py
│   ├── test_model.py
│   ├── test_plot.py
│   ├── test_simclr.py
│   └── test_supervised.py
├── Makefile
└── pyproject.toml
```

---

## Make Targets

### Setup
```bash
make sync             # uv sync --dev
```

### Data  (streamed — tar.gz never written to disk)
```bash
make smoke-data       # train split only (~120 MB on disk) — safe for Codespaces
make data             # train + test splits (~310 MB on disk) — supervised baseline + eval
make data-unlabeled   # unlabeled split (~2.5 GB on disk) — required for SimCLR pretraining
```

### Training
```bash
make pretrain         # SimCLR pretraining on unlabelled data
make supervised       # supervised baseline (all 5000 labels)
make fixmatch         # FixMatch (labels_per_class from config)
make ssl_fixmatch     # SimCLR pretrain + FixMatch fine-tune
make sweep            # FixMatch across all label fractions (4, 10, 25, 100, 500/class)
make smoke            # fast 2-epoch smoke run on small data subset (Codespaces-safe)
```

### Evaluation & export
```bash
make evaluate         # evaluate all checkpoints on test set
make plot             # generate label efficiency curve → plots/label_efficiency.png
make export           # export best model to ONNX
```

### Quality
```bash
make test             # pytest with --cov (currently 81% coverage, 58 tests)
make lint             # ruff format → ruff check --fix → ty check
```

### Cleanup
```bash
make clean            # remove data/stl10_binary/, checkpoints/*.pt, plots/*.png, mlruns/
```

### Pipelines
```bash
make all              # data + data-unlabeled → pretrain → sweep → plot
make kaggle_push      # push notebooks to Kaggle for GPU training
```

---

## Compute Environments

| Environment | Use |
|---|---|
| **GitHub Codespaces** (2-core CPU) | Development, `make test`, `make lint`, `make smoke` |
| **Kaggle** (T4/P100 GPU) | Full training — SimCLR 100 epochs + FixMatch 5-fraction sweep |

- On Codespaces, `make smoke-data && make smoke` is the safe workflow (no 2.5 GB download).
- `uv sync` installs CPU-only PyTorch wheels (~1.5 GB venv vs ~5 GB with CUDA). Kaggle notebooks use pip and already have CUDA torch pre-installed, so they are unaffected.
- Kaggle notebooks have a `SMOKE_TEST = False` flag at the top — set to `True` for low-disk runs.

---

## Architecture

### Backbone: ResNet-18
- Input: 64×64 RGB (downsampled from 96×96 for efficiency)
- Random init — no ImageNet weights (isolates SSL value)
- 512-dimensional feature vector before final FC

### SimCLR Projection Head
```
ResNet-18 encoder (512d)
    -> Linear(512, 512) -> BatchNorm -> ReLU
    -> Linear(512, 128)    # projection head output z
```
Projection head is discarded after pretraining.

### FixMatch Classifier
```
ResNet-18 encoder (512d, SimCLR-pretrained or random init)
    -> Linear(512, 10)
```

### Augmentation Pipelines
- **SimCLR**: random crop+resize, flip, color jitter, grayscale, Gaussian blur — applied twice for two views
- **FixMatch weak**: random crop+resize, flip
- **FixMatch strong**: RandAugment (AutoContrast, Equalize, Rotate, Solarize, Posterize, Color, Contrast, Brightness, Sharpness, ShearX/Y, TranslateX/Y)

---

## Module API Reference

### `config.py`
```python
def load_simclr_config(path)    -> SimCLRConfig
def load_fixmatch_config(path)  -> FixMatchConfig
def load_supervised_config(path) -> SupervisedConfig
```

### `dataset.py`
```python
def sample_label_fraction(
    dataset: Dataset[Any], labels_per_class: int, num_classes: int, seed: int,
) -> tuple[Subset[Any], Subset[Any]]

def get_stl10_splits(
    config, labels_per_class, seed, input_size=64,
    smoke_test=False, max_labelled=50, max_unlabelled=1000,
) -> tuple[DataLoader, DataLoader, DataLoader]
# Returns: labelled_loader, unlabelled_loader, test_loader

def get_unlabelled_loader(...) -> DataLoader   # for SimCLR pretraining (uses split="unlabeled")
```

### `model.py`
```python
class ResNet18WithProjection(nn.Module)   # SimCLR: encoder + projection head
class ResNet18Classifier(nn.Module)       # FixMatch/supervised: encoder + linear head
class EMAModel                            # exponential moving average of weights
```

### `simclr.py`
```python
class NTXentLoss(nn.Module)
class SimCLRDataset(Dataset[tuple[Tensor, Tensor]])   # two augmented views
def train_simclr(model, loader, config, checkpoint_dir, smoke_test) -> ResNet18WithProjection
```

### `fixmatch.py`
```python
class PseudoLabelFilter:
    def filter(self, probs) -> tuple[mask, pseudo_labels]
class FixMatchUnlabelledDataset(Dataset[tuple[Tensor, Tensor]])  # (weak, strong) views
def train_fixmatch(model, labelled_loader, unlabelled_loader, config, ...) -> ResNet18Classifier
```

### `evaluate.py`
```python
@dataclass
class EvalResult:
    accuracy: float
    top5_accuracy: float
    confusion_matrix: np.ndarray
    per_class_accuracy: dict[str, float]

def evaluate(model, test_loader, class_names=None) -> EvalResult
def plot_tsne_embeddings(model, loader, save_path, max_samples=2000) -> None
```

### `export.py`
```python
def export_onnx(checkpoint, output, num_classes=10, input_size=64) -> None
# Exports + validates with onnx.checker and onnxruntime (asserts max diff < 1e-4)
```

### `plot.py`
```python
def plot_label_efficiency_curve(results: dict[str, dict[int, float]], save_path, title) -> None
def plot_training_curves(train_losses, val_accs=None, save_path, title) -> None
```

---

## Config Schema

### `config/simclr.yaml`
```yaml
model:
  backbone: "resnet18"
  projection_dim: 128
  input_size: 64

training:
  epochs: 100
  batch_size: 256
  temperature: 0.5
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 10

data:
  dataset: "stl10"
  data_dir: "data/"
  num_workers: 2

smoke_test:
  enabled: false        # set true on Codespaces
  max_labelled: 50
  max_unlabelled: 1000
  max_epochs: 2
```

### `config/fixmatch.yaml`
```yaml
model:
  backbone: "resnet18"
  pretrained_simclr: null   # path to SimCLR checkpoint, or null
  input_size: 64
  num_classes: 10

training:
  epochs: 200
  batch_size_labelled: 64
  batch_size_unlabelled: 192  # 1:3 labelled:unlabelled ratio
  learning_rate: 3.0e-2
  weight_decay: 5.0e-4
  momentum: 0.9
  ema_decay: 0.999
  threshold: 0.95             # pseudo-label confidence threshold
  lambda_u: 1.0

data:
  dataset: "stl10"
  data_dir: "data/"
  labels_per_class: 100       # 4 | 10 | 25 | 100 | 500
  random_seed: 42
  num_workers: 2
```

---

## Engineering Conventions

| Convention | Choice |
|---|---|
| Package manager | `uv` with `pyproject.toml` |
| Layout | `src/` layout |
| Config | Per-experiment YAML, validated by Pydantic v2 |
| Typing | Fully typed Python (`ty` strict) |
| Testing | `pytest` + `pytest-cov`; synthetic fixtures — no real data in CI |
| Logging | `loguru` |
| Experiment tracking | MLflow (local) |
| CI | GitHub Actions: `make sync` → `make lint` → `make test` |
| License | Apache 2.0 |

### Key Dependencies
```toml
# Runtime
torch = ">=2.2"
torchvision = ">=0.17"
numpy = ">=1.26"
scikit-learn = ">=1.4"
matplotlib = ">=3.8"
onnx = ">=1.16"
onnxruntime = ">=1.17"
onnxscript = ">=0.1"
mlflow = ">=2.12"
pydantic = ">=2.6"
loguru = ">=0.7"
pyyaml = ">=6.0"

# Dev
pytest = ">=8.0"
pytest-cov = ">=5.0"
ty = ">=0.0.1a6"
ruff = ">=0.4"
```

---

## Expected Results

| Method | Labels / class | Expected accuracy |
|---|---|---|
| Supervised baseline | 500 (full) | ~72% |
| FixMatch (random init) | 100 | ~68% |
| FixMatch (random init) | 25 | ~58% |
| SimCLR + FixMatch | 100 | ~75% |
| SimCLR + FixMatch | 25 | ~70% |
| SimCLR + FixMatch | 4 | ~60% |

Full training on Kaggle T4: supervised baseline ~20 min; SimCLR + FixMatch sweep ~90 min.

---

## Known Risks

| Risk | Mitigation |
|---|---|
| SimCLR too slow on 2 CPUs | Use `make smoke` (2 epochs, 1k images) |
| FixMatch unstable with 4 labels/class | EMA model for pseudo-labels; supervised warmup |
| STL-10 download fills Codespaces disk | Streaming download (tar.gz never on disk); `make smoke-data` writes ~120 MB; unlabeled split on Kaggle only |
| Kaggle session timeout (9h) | Two separate notebooks; checkpoints saved to Kaggle output |
| Reproducibility across seeds | Fix seed in config; report mean over 3 seeds |
