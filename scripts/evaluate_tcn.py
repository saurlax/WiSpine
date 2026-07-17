from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from train_tcn import build_arrays, make_loader, predict

from wispine.features import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    CSISample,
    list_csi_samples,
)
from wispine.models import TCNClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved TCN model without training.")
    parser.add_argument("--model-dir", default="outputs/tcn_phase_ratio")
    parser.add_argument("--data-dir", default=None, help="Override dataset root for split paths.")
    parser.add_argument("--cache-dir", default=None, help="Override cache directory.")
    parser.add_argument("--split", choices=("test", "val", "train", "all"), default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-json", default=None, help="Optional path for evaluation results.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-windows-per-file", type=int, default=None)
    parser.add_argument("--reference-rx", type=int, default=None)
    parser.add_argument(
        "--scaled",
        action="store_true",
        help="Override training args to use scaled CSI.",
    )
    parser.add_argument(
        "--no-unwrap",
        action="store_true",
        help="Override training args to disable unwrap.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_arg(metrics: dict[str, Any], name: str, fallback: Any = None) -> Any:
    return metrics.get("args", {}).get(name, fallback)


def split_sample_from_json(row: dict[str, Any], data_dir: Path | None) -> CSISample:
    path = Path(row["path"])
    if data_dir is not None:
        path = data_dir / row["class_name"] / path.name
    return CSISample(path=path, label=int(row["label"]), class_name=str(row["class_name"]))


def load_samples(args: argparse.Namespace, metrics: dict[str, Any]) -> list[CSISample]:
    data_dir = Path(args.data_dir) if args.data_dir is not None else None
    if args.split == "all":
        root = data_dir if data_dir is not None else Path(get_arg(metrics, "data_dir", "data"))
        return list_csi_samples(root)

    split_rows = metrics.get("split", {}).get(args.split)
    if not split_rows:
        raise ValueError(f"metrics.json does not contain split data for {args.split!r}.")
    return [split_sample_from_json(row, data_dir) for row in split_rows]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    metrics = load_json(model_dir / "metrics.json")
    samples = load_samples(args, metrics)

    cache_dir_value = (
        args.cache_dir if args.cache_dir is not None else get_arg(metrics, "cache_dir")
    )
    cache_dir = Path(cache_dir_value) if cache_dir_value else None
    window_size = (
        args.window_size
        if args.window_size is not None
        else get_arg(metrics, "window_size", 512)
    )
    stride = args.stride if args.stride is not None else get_arg(metrics, "stride", 256)
    max_windows_per_file = (
        args.max_windows_per_file
        if args.max_windows_per_file is not None
        else get_arg(metrics, "max_windows_per_file")
    )
    reference_rx = (
        args.reference_rx
        if args.reference_rx is not None
        else get_arg(metrics, "reference_rx", 0)
    )
    scaled = args.scaled or bool(get_arg(metrics, "scaled", False))
    unwrap = not (args.no_unwrap or bool(get_arg(metrics, "no_unwrap", False)))
    batch_size = (
        args.batch_size if args.batch_size is not None else get_arg(metrics, "batch_size", 32)
    )

    x, y = build_arrays(
        samples,
        window_size=window_size,
        stride=stride,
        max_windows_per_file=max_windows_per_file,
        reference_rx=reference_rx,
        scaled=scaled,
        unwrap=unwrap,
        cache_dir=cache_dir,
    )
    normalization = np.load(model_dir / "normalization.npz")
    x = (x - normalization["mean"]) / np.maximum(normalization["std"], 1e-6)

    loader = make_loader(x, y, batch_size=batch_size, shuffle=False)
    device = resolve_device(args.device)
    model = TCNClassifier(
        input_features=int(metrics.get("input_features", x.shape[-1])),
        num_classes=len(CLASS_TO_INDEX),
        dropout=float(get_arg(metrics, "dropout", 0.2)),
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))

    true_labels, predicted_labels = predict(model, loader, device=device)
    target_names = [INDEX_TO_CLASS[index] for index in range(len(INDEX_TO_CLASS))]
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels).tolist()
    result = {
        "model_dir": str(model_dir),
        "split": args.split,
        "device": str(device),
        "sample_count": len(samples),
        "window_count": int(x.shape[0]),
        "test_report": report,
        "confusion_matrix": matrix,
    }

    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
        print(f"saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()