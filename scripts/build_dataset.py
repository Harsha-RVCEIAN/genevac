"""Inventory and prepare real-world ETH/UCY or SDD trajectory data."""
import argparse
import json
from pathlib import Path

from genevac.prediction.dataset_pipeline import (
    SUPPORTED_DATASETS,
    export_dataset,
    inventory_datasets,
    load_eth_ucy,
    load_sdd_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output", default="data/processed/real_world")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="ETH")
    parser.add_argument("--file")
    parser.add_argument("--scene", default="default")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--source-split", choices=("train", "validation", "test"))
    parser.add_argument("--seconds-per-source-frame", type=float, default=0.04)
    parser.add_argument("--sampling-stride", type=int, default=1)
    parser.add_argument(
        "--sdd-seconds-per-frame",
        type=float,
        help="Required for SDD tables using frame indices instead of timestamps",
    )
    parser.add_argument(
        "--sdd-coordinates-in-metres",
        action="store_true",
        help="Explicitly confirm SDD x/y were homography-transformed into world metres",
    )
    args = parser.parse_args()

    if args.inventory_only:
        print(json.dumps(inventory_datasets(args.raw_root), indent=2))
        return 0
    if not args.file:
        parser.error("--file is required unless --inventory-only is used")

    source = Path(args.file)
    if args.dataset == "SDD":
        tracks = load_sdd_table(
            source,
            seconds_per_frame=args.sdd_seconds_per_frame,
            coordinates_in_metres=args.sdd_coordinates_in_metres,
        )
    else:
        tracks = load_eth_ucy(
            source,
            seconds_per_source_frame=args.seconds_per_source_frame,
            sampling_stride=args.sampling_stride,
        )

    manifest = export_dataset(
        tracks,
        args.output,
        dataset=args.dataset,
        scene=args.scene,
        source=source,
        fixed_split=args.source_split,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
