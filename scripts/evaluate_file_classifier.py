from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from train_file_classifier import build_file_features, evaluate

from wispine.dataset import sample_to_json
from wispine.features import CSISample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved file-level CSI classifier.")
    parser.add_argument("--model-dir", default="outputs/file_phase_ratio")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    train_args = metrics["args"]
    data_dir = Path(args.data_dir) if args.data_dir else None
    samples = []
    for row in metrics["split"][args.split]:
        path = Path(row["path"])
        if data_dir is not None:
            path = data_dir / row["class_name"] / path.name
        samples.append(
            CSISample(path=path, label=int(row["label"]), class_name=row["class_name"])
        )

    cache_value = args.cache_dir if args.cache_dir is not None else train_args["cache_dir"]
    x, y = build_file_features(
        samples,
        window_size=int(train_args["window_size"]),
        stride=int(train_args["stride"]),
        max_windows_per_file=train_args["max_windows_per_file"],
        reference_rx=int(train_args["reference_rx"]),
        scaled=bool(train_args["scaled"]),
        unwrap=not bool(train_args["no_unwrap"]),
        cache_dir=Path(cache_value) if cache_value else None,
    )
    result = {
        "model_dir": str(model_dir),
        "split": args.split,
        "evaluation_scope": metrics["evaluation_scope"],
        "samples": [sample_to_json(sample) for sample in samples],
        **evaluate(joblib.load(model_dir / "model.joblib"), x, y),
    }
    output_path = (
        Path(args.output_json)
        if args.output_json
        else model_dir / f"evaluation_{args.split}.json"
    )
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
