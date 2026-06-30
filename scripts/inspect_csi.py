from __future__ import annotations

import argparse
from pathlib import Path

from wispine.csi import read_amplitude_phase, read_phase_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a CSI .dat file.")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/kyphosis/cyd-k01.dat",
        help="Path to an Intel 5300 .dat CSI file.",
    )
    parser.add_argument("--scaled", action="store_true", help="Use CSIKit scaled CSI values.")
    parser.add_argument(
        "--phase-ratio",
        action="store_true",
        help="Also inspect antenna phase-ratio features angle(H_rx / H_ref).",
    )
    parser.add_argument(
        "--reference-rx",
        type=int,
        default=0,
        help="Reference receive antenna index for --phase-ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    metrics = read_amplitude_phase(path, scaled=args.scaled)

    print(f"file: {path}")
    print(f"frames reported by reader: {metrics.frame_count}")
    print(f"subcarriers: {metrics.subcarrier_count}")
    print(f"timestamps: {metrics.timestamps.shape[0]}")
    print(f"amplitude shape: {metrics.amplitude.shape}")
    print(f"phase shape: {metrics.phase.shape}")
    print(
        "amplitude min/max/mean: "
        f"{metrics.amplitude.min():.6f} / {metrics.amplitude.max():.6f} / "
        f"{metrics.amplitude.mean():.6f}"
    )
    print(
        "phase min/max/mean: "
        f"{metrics.phase.min():.6f} / {metrics.phase.max():.6f} / "
        f"{metrics.phase.mean():.6f}"
    )

    if args.phase_ratio:
        ratio = read_phase_ratio(path, reference_rx=args.reference_rx, scaled=args.scaled)
        print(f"phase-ratio reference rx: {ratio.reference_rx}")
        print(f"phase-ratio compared rx: {ratio.compared_rx}")
        print(f"phase-ratio shape: {ratio.phase_ratio.shape}")
        print(
            "phase-ratio min/max/mean: "
            f"{ratio.phase_ratio.min():.6f} / {ratio.phase_ratio.max():.6f} / "
            f"{ratio.phase_ratio.mean():.6f}"
        )


if __name__ == "__main__":
    main()
