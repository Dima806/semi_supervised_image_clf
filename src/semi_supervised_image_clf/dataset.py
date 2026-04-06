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

import torch
from loguru import logger
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

    Args:
        config: data config containing ``data_dir`` and ``num_workers``.
        labels_per_class: number of labelled images per class (4/10/25/100/500).
        seed: random seed for reproducible label sampling.
        input_size: spatial resolution to resize images to.
        smoke_test: if True, cap dataset sizes for fast iteration.
        max_labelled: smoke-test cap on labelled images.
        max_unlabelled: smoke-test cap on unlabelled images.
    """
    data_dir = config.data_dir
    num_workers = config.num_workers
    transform = base_transform(input_size)

    # STL-10 has two labelled splits: 'train' (5000) and 'test' (8000)
    # The 'unlabeled' split contains 100k images with label=-1
    labelled_full = datasets.STL10(
        root=data_dir, split="train", transform=transform, download=False
    )
    unlabelled_full = datasets.STL10(
        root=data_dir, split="unlabeled", transform=transform, download=False
    )
    test_dataset = datasets.STL10(root=data_dir, split="test", transform=transform, download=False)

    num_classes = 10

    # Sample label fraction
    labelled_subset, extra_unlabelled = sample_label_fraction(
        labelled_full, labels_per_class, num_classes, seed
    )

    # Combine original unlabelled + leftovers from labelled split
    unlabelled_dataset: Dataset = torch.utils.data.ConcatDataset(  # type: ignore[type-arg]
        [unlabelled_full, extra_unlabelled]
    )

    if smoke_test:
        labelled_subset = Subset(
            labelled_subset, list(range(min(max_labelled, len(labelled_subset))))
        )
        unlabelled_dataset = Subset(
            unlabelled_dataset, list(range(min(max_unlabelled, len(unlabelled_dataset))))
        )  # type: ignore[arg-type]
        logger.info(
            f"Smoke test: {len(labelled_subset)} labelled, {len(unlabelled_dataset)} unlabelled"
        )  # type: ignore[arg-type]

    logger.info(
        f"Dataset splits — labelled: {len(labelled_subset)}, "  # type: ignore[arg-type]
        f"unlabelled: {len(unlabelled_dataset)}, test: {len(test_dataset)}"  # type: ignore[arg-type]
    )

    labelled_loader = DataLoader(
        labelled_subset,
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
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
    """Return a DataLoader over the full STL-10 unlabeled split for SimCLR."""
    transform = base_transform(input_size)
    dataset = datasets.STL10(root=data_dir, split="unlabeled", transform=transform, download=False)
    if smoke_test:
        dataset = Subset(dataset, list(range(min(max_unlabelled, len(dataset)))))  # type: ignore[assignment]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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
