from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wispine.features import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    CSISample,
    list_csi_samples,
    load_phase_ratio_sequence,
    make_windows,
)
from wispine.models import TCNClassifier


@dataclass(frozen=True)
class DatasetSplit:
    train: list[CSISample]
    val: list[CSISample]
    test: list[CSISample]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TCN on phase-ratio CSI features.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Dataset root with class subdirectories.",
    )
    parser.add_argument("--output-dir", default="outputs/tcn_phase_ratio", help="Output directory.")
    parser.add_argument(
        "--cache-dir",
        default="artifacts/phase_ratio_windows",
        help="Directory for cached window arrays.",
    )
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-windows-per-file", type=int, default=None)
    parser.add_argument("--max-files-per-class", type=int, default=None)
    parser.add_argument("--reference-rx", type=int, default=0)
    parser.add_argument("--scaled", action="store_true", help="Use CSIKit scaled CSI values.")
    parser.add_argument("--no-unwrap", action="store_true", help="Disable phase unwrap over time.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_samples(
    samples: list[CSISample],
    *,
    test_size: float,
    val_size: float,
    seed: int,
) -> DatasetSplit:
    labels = [sample.label for sample in samples]
    class_count = len(set(labels))
    test_size = max(test_size, class_count / len(samples))
    train_val, test = train_test_split(
        samples,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [sample.label for sample in train_val]
    val_size = max(val_size, class_count / len(samples))
    relative_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val_labels,
    )
    return DatasetSplit(train=train, val=val, test=test)


def build_arrays(
    samples: list[CSISample],
    *,
    window_size: int,
    stride: int,
    max_windows_per_file: int | None,
    reference_rx: int,
    scaled: bool,
    unwrap: bool,
    cache_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for index, sample in enumerate(samples, start=1):
        cache_path = make_cache_path(
            sample,
            cache_dir=cache_dir,
            window_size=window_size,
            stride=stride,
            max_windows_per_file=max_windows_per_file,
            reference_rx=reference_rx,
            scaled=scaled,
            unwrap=unwrap,
        )
        if cache_path is not None and cache_path.exists():
            print(f"[{index:03d}/{len(samples):03d}] loading cache {cache_path}")
            windows = np.load(cache_path)["windows"]
        else:
            print(f"[{index:03d}/{len(samples):03d}] extracting {sample.path}")
            sequence = load_phase_ratio_sequence(
                sample.path,
                reference_rx=reference_rx,
                scaled=scaled,
                unwrap=unwrap,
            )
            windows = make_windows(
                sequence,
                window_size=window_size,
                stride=stride,
                max_windows=max_windows_per_file,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, windows=windows)
        labels = np.full((windows.shape[0],), sample.label, dtype=np.int64)
        all_windows.append(windows)
        all_labels.append(labels)

    return np.concatenate(all_windows), np.concatenate(all_labels)


def make_cache_path(
    sample: CSISample,
    *,
    cache_dir: Path | None,
    window_size: int,
    stride: int,
    max_windows_per_file: int | None,
    reference_rx: int,
    scaled: bool,
    unwrap: bool,
) -> Path | None:
    if cache_dir is None:
        return None

    key = "|".join(
        [
            str(sample.path.resolve()),
            str(sample.path.stat().st_mtime_ns),
            str(window_size),
            str(stride),
            str(max_windows_per_file),
            str(reference_rx),
            str(scaled),
            str(unwrap),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / sample.class_name / f"{sample.path.stem}-{digest}.npz"


def standardize(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)

    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (test_x - mean) / std,
        mean.squeeze(),
        std.squeeze(),
    )


def make_loader(x: np.ndarray, y: np.ndarray, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def sample_to_json(sample: CSISample) -> dict[str, str | int]:
    return {
        "path": str(sample.path),
        "label": sample.label,
        "class_name": sample.class_name,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    predictions: list[int] = []
    labels: list[int] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
            labels.extend(batch_y.detach().cpu().tolist())

    return total_loss / len(labels), accuracy_score(labels, predictions)


def predict(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(batch_y.tolist())

    return np.asarray(labels), np.asarray(predictions)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    samples = list_csi_samples(args.data_dir, max_files_per_class=args.max_files_per_class)
    split = split_samples(
        samples,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    train_x, train_y = build_arrays(
        split.train,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_file=args.max_windows_per_file,
        reference_rx=args.reference_rx,
        scaled=args.scaled,
        unwrap=not args.no_unwrap,
        cache_dir=cache_dir,
    )
    val_x, val_y = build_arrays(
        split.val,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_file=args.max_windows_per_file,
        reference_rx=args.reference_rx,
        scaled=args.scaled,
        unwrap=not args.no_unwrap,
        cache_dir=cache_dir,
    )
    test_x, test_y = build_arrays(
        split.test,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_file=args.max_windows_per_file,
        reference_rx=args.reference_rx,
        scaled=args.scaled,
        unwrap=not args.no_unwrap,
        cache_dir=cache_dir,
    )

    train_x, val_x, test_x, mean, std = standardize(train_x, val_x, test_x)
    np.savez(output_dir / "normalization.npz", mean=mean, std=std)

    train_loader = make_loader(train_x, train_y, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_x, test_y, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNClassifier(
        input_features=train_x.shape[-1],
        num_classes=len(CLASS_TO_INDEX),
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion=criterion, device=device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    true_labels, predicted_labels = predict(model, test_loader, device=device)

    target_names = [INDEX_TO_CLASS[index] for index in range(len(INDEX_TO_CLASS))]
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels).tolist()

    metadata = {
        "args": vars(args),
        "classes": CLASS_TO_INDEX,
        "device": str(device),
        "input_features": int(train_x.shape[-1]),
        "split": {
            "train": [sample_to_json(sample) for sample in split.train],
            "val": [sample_to_json(sample) for sample in split.val],
            "test": [sample_to_json(sample) for sample in split.test],
        },
        "history": history,
        "test_report": report,
        "confusion_matrix": matrix,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(json.dumps(report, indent=2))
    print(f"saved best model and metrics to {output_dir}")


if __name__ == "__main__":
    main()
