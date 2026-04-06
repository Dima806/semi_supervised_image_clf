"""STL-10 dataset loading and label-fraction sampling."""

from __future__ import annotations

import argparse
import random
import tarfile
import urllib.request
from collections import defaultdict
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from semi_supervised_image_clf.config import DataConfig, FixMatchDataConfig

# ---------------------------------------------------------------------------
# STL-10 download constants
# ---------------------------------------------------------------------------

_STL10_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
_STL10_BASE = "stl10_binary"

# Files required to load each split via torchvision.
# Torchvision's split name is "unlabeled" (US spelling).
_SPLIT_FILES: dict[str, list[str]] = {
    "train": ["train_X.bin", "train_y.bin", "class_names.txt", "fold_indices.txt"],
    "test": ["test_X.bin", "test_y.bin"],
    "unlabeled": ["unlabeled_X.bin"],
}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def base_transform(input_size: int = 64) -> transforms.Compose:
    """Minimal transform: resize + to-tensor + normalise."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ---------------------------------------------------------------------------
# Minimal STL-10 binary reader
# ---------------------------------------------------------------------------

# torchvision's STL10._check_integrity requires ALL five binary files to be
# present (train_X, train_y, unlabeled_X, test_X, test_y) even when only one
# split is requested.  Our tiered download intentionally omits unused splits,
# so the integrity check always fails unless everything is present.
# _STL10Split reads only the files it needs, skipping the global check.

_STL10_SPLIT_FILES: dict[str, tuple[str, str | None]] = {
    "train": ("train_X.bin", "train_y.bin"),
    "test": ("test_X.bin", "test_y.bin"),
    "unlabeled": ("unlabeled_X.bin", None),
}
_STL10_IMG_SHAPE = (96, 96, 3)  # HWC after transpose


class _STL10Split(Dataset[tuple[Any, int]]):
    """Read one STL-10 binary split without a global integrity check."""

    def __init__(
        self,
        root: str,
        split: str,
        transform: transforms.Compose | None = None,
    ) -> None:
        base = Path(root) / _STL10_BASE
        data_file, label_file = _STL10_SPLIT_FILES[split]

        raw = np.fromfile(base / data_file, dtype=np.uint8)
        images_chw = raw.reshape(-1, 3, 96, 96)
        self._images: np.ndarray = np.transpose(images_chw, (0, 2, 3, 1))  # NHWC

        if label_file is not None:
            self._labels: np.ndarray = (
                np.fromfile(base / label_file, dtype=np.uint8).astype(np.int64) - 1
            )
        else:
            self._labels = np.full(len(self._images), -1, dtype=np.int64)

        self.transform = transform

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: Any) -> tuple[Any, int]:
        img = Image.fromarray(self._images[int(index)])
        label = int(self._labels[int(index)])
        if self.transform is not None:
            return self.transform(img), label
        return img, label


# ---------------------------------------------------------------------------
# Synthetic dataset for smoke tests (no real data required)
# ---------------------------------------------------------------------------


class _SyntheticDataset(Dataset[tuple[Any, int]]):
    """Random-pixel PIL-image dataset for smoke tests.

    Returns ``(PIL.Image, label)`` by default — compatible with torchvision
    augmentation pipelines that expect raw images.  Pass ``transform`` to get
    normalised tensors instead (used for the test/eval path).
    """

    def __init__(
        self,
        n_samples: int = 200,
        input_size: int = 64,
        num_classes: int = 10,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.input_size = input_size
        self.num_classes = num_classes
        self.transform = transform
        rng = np.random.default_rng(0)
        self._imgs: np.ndarray = rng.integers(
            0, 255, (n_samples, input_size, input_size, 3), dtype=np.uint8
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: Any) -> tuple[Any, int]:
        img = Image.fromarray(self._imgs[int(index) % self.n_samples])
        label = int(index) % self.num_classes
        if self.transform is not None:
            return self.transform(img), label
        return img, label


# ---------------------------------------------------------------------------
# Label-fraction sampling
# ---------------------------------------------------------------------------


def sample_label_fraction(
    dataset: Dataset[Any],
    labels_per_class: int,
    num_classes: int,
    seed: int,
) -> tuple[Subset[Any], Subset[Any]]:
    """Split a labelled dataset into a small labelled subset and the rest.

    Returns:
        labelled_subset: ``labels_per_class * num_classes`` images.
        unlabelled_subset: remaining images (labels stripped at training time).
    """
    rng = random.Random(seed)

    # Group indices by class
    class_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(cast(Sized, dataset))):
        _, label = dataset[idx]
        class_indices[int(label)].append(idx)

    labelled_indices: list[int] = []
    unlabelled_indices: list[int] = []

    for cls in range(num_classes):
        indices = class_indices[cls]
        rng.shuffle(indices)
        labelled_indices.extend(indices[:labels_per_class])
        unlabelled_indices.extend(indices[labels_per_class:])

    return Subset(dataset, labelled_indices), Subset(dataset, unlabelled_indices)


# ---------------------------------------------------------------------------
# Main split builder
# ---------------------------------------------------------------------------


def get_stl10_splits(
    config: FixMatchDataConfig | DataConfig,
    labels_per_class: int,
    seed: int,
    input_size: int = 64,
    smoke_test: bool = False,
    max_labelled: int = 50,
    max_unlabelled: int = 1000,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (labelled_loader, unlabelled_loader, test_loader) for STL-10.

    Design rule
    -----------
    * Labelled and test loaders serve **normalised tensors** (``base_transform``
      applied) — they go directly into the training/eval loss.
    * The unlabelled loader serves **raw PIL images** (no transform) — it is
      always wrapped by ``FixMatchUnlabelledDataset`` which applies its own
      augmentation pipeline (``WeakAugmentation`` / ``StrongAugmentation``),
      both of which include ``ToTensor``.  Applying ``base_transform`` here
      would cause a double-transform error.

    Args:
        config: data config containing ``data_dir`` and ``num_workers``.
        labels_per_class: number of labelled images per class (4/10/25/100/500).
        seed: random seed for reproducible label sampling.
        input_size: spatial resolution to resize images to.
        smoke_test: if True, use synthetic data for missing splits and cap sizes.
        max_labelled: smoke-test cap on labelled images.
        max_unlabelled: smoke-test cap on unlabelled images.
    """
    data_dir = config.data_dir
    num_workers = config.num_workers
    transform = base_transform(input_size)
    num_classes = 10

    # ------------------------------------------------------------------
    # Train split — loaded twice:
    #   • with base_transform  → for the labelled loader (tensors)
    #   • without transform    → for extra_unlabelled fed into FixMatch
    # Both use the same seed so sample_label_fraction picks identical indices.
    # ------------------------------------------------------------------
    train_transformed = _STL10Split(root=data_dir, split="train", transform=transform)
    train_raw = _STL10Split(root=data_dir, split="train", transform=None)

    labelled_subset_transformed, _ = sample_label_fraction(
        train_transformed, labels_per_class, num_classes, seed
    )
    _, extra_unlabelled_raw = sample_label_fraction(train_raw, labels_per_class, num_classes, seed)

    # ------------------------------------------------------------------
    # Unlabelled split (PIL images — no transform)
    # ------------------------------------------------------------------
    unlabelled_full: Dataset[Any]
    if smoke_test and not _files_present(data_dir, ["unlabeled"]):
        unlabelled_full = _SyntheticDataset(n_samples=max_unlabelled, input_size=input_size)
    else:
        unlabelled_full = _STL10Split(root=data_dir, split="unlabeled", transform=None)

    # ------------------------------------------------------------------
    # Test split (tensors via base_transform)
    # ------------------------------------------------------------------
    test_dataset: Dataset[Any]
    if smoke_test and not _files_present(data_dir, ["test"]):
        test_dataset = _SyntheticDataset(
            n_samples=max_unlabelled, input_size=input_size, transform=transform
        )
    else:
        test_dataset = _STL10Split(root=data_dir, split="test", transform=transform)

    # Combine full unlabelled pool with leftover train images (both PIL)
    unlabelled_dataset: Dataset[Any] = torch.utils.data.ConcatDataset(  # type: ignore[type-arg]
        [unlabelled_full, extra_unlabelled_raw]
    )

    if smoke_test:
        labelled_subset_transformed = Subset(
            labelled_subset_transformed,
            list(range(min(max_labelled, len(labelled_subset_transformed)))),
        )
        unlabelled_dataset = Subset(
            unlabelled_dataset,
            list(range(min(max_unlabelled, len(unlabelled_dataset)))),
        )  # type: ignore[arg-type]
        logger.info(
            f"Smoke test: {len(labelled_subset_transformed)} labelled, "
            f"{len(unlabelled_dataset)} unlabelled"  # type: ignore[arg-type]
        )

    logger.info(
        f"Dataset splits — labelled: {len(labelled_subset_transformed)}, "
        f"unlabelled: {len(unlabelled_dataset)}, test: {len(test_dataset)}"  # type: ignore[arg-type]
    )

    n_labelled = len(labelled_subset_transformed)
    labelled_bs = min(64, n_labelled)
    labelled_loader = DataLoader(
        labelled_subset_transformed,
        batch_size=labelled_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=n_labelled >= 64,
    )
    unlabelled_loader = DataLoader(
        unlabelled_dataset,  # type: ignore[arg-type]
        batch_size=192,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return labelled_loader, unlabelled_loader, test_loader


def get_unlabelled_loader(
    data_dir: str,
    input_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 2,
    smoke_test: bool = False,
    max_unlabelled: int = 1000,
) -> DataLoader:  # type: ignore[type-arg]
    """Return a DataLoader over the STL-10 unlabeled split for SimCLR.

    Returns raw PIL images (no transform) because the dataset is immediately
    wrapped by ``SimCLRDataset`` which applies ``SimCLRAugmentation``
    (including ``ToTensor``).  In smoke-test mode synthetic PIL images are
    used so that no download is required.
    """
    if smoke_test:
        dataset: Dataset[Any] = _SyntheticDataset(n_samples=max_unlabelled, input_size=input_size)
    else:
        dataset = _STL10Split(root=data_dir, split="unlabeled", transform=None)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if smoke_test else num_workers,
        pin_memory=not smoke_test,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# CLI: streaming download — no tar.gz written to disk
# ---------------------------------------------------------------------------


def _files_present(data_dir: str, splits: list[str]) -> bool:
    """Return True if all binary files for the requested splits already exist."""
    base = Path(data_dir) / _STL10_BASE
    for split in splits:
        for fname in _SPLIT_FILES.get(split, []):
            if not (base / fname).exists():
                return False
    return True


def _download(data_dir: str, splits: list[str] | None = None) -> None:
    """Stream the STL-10 archive and extract only the files needed for *splits*.

    The tar.gz is never written to disk: data is extracted directly from the
    HTTP response stream.  Skipped files (e.g. ``unlabeled_X.bin`` when only
    downloading the train split) are drained from the stream without being
    stored, so disk usage equals only the extracted binary files.
    """
    if splits is None:
        splits = ["train", "test", "unlabeled"]

    if _files_present(data_dir, splits):
        logger.info("STL-10 data already present, skipping download.")
        _log_split_sizes(data_dir, splits)
        return

    # Build the set of archive member names we want to keep.
    wanted: set[str] = set()
    for split in splits:
        for fname in _SPLIT_FILES.get(split, []):
            wanted.add(f"{_STL10_BASE}/{fname}")

    base_dir = Path(data_dir) / _STL10_BASE
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Streaming STL-10 for splits {splits} from {_STL10_URL}")
    logger.info("The full archive is streamed; only requested files are written to disk.")

    with (
        urllib.request.urlopen(_STL10_URL) as response,
        tarfile.open(fileobj=response, mode="r|gz") as tar,
    ):
        for member in tar:
            if not member.isfile():
                continue
            if member.name in wanted:
                fname = Path(member.name).name
                out_path = base_dir / fname
                logger.info(f"  Extracting {fname} ({member.size / 1e6:.0f} MB) ...")
                f = tar.extractfile(member)
                if f is not None:
                    out_path.write_bytes(f.read())
            # Members not in `wanted` are automatically skipped (drained
            # from the stream by tarfile without being stored).

    logger.info("Download complete.")
    _log_split_sizes(data_dir, splits)


def _log_split_sizes(data_dir: str, splits: list[str]) -> None:
    for split in splits:
        try:
            ds = datasets.STL10(root=data_dir, split=split, download=False)
            logger.info(f"  {split}: {len(ds):,} images")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "unlabeled"],
        choices=["train", "test", "unlabeled"],
        help="Which STL-10 splits to download (default: all three)",
    )
    args = parser.parse_args()
    if args.download:
        _download(args.data_dir, args.splits)
