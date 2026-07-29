from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.experiments import METRIC_COLUMNS
from benchmark.telecomts import PAPER_CACHE_SHA256, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare generated metrics with the paper artifacts",
    )
    parser.add_argument(
        "--expected-dir",
        type=Path,
        default=Path("artifacts/expected"),
    )
    parser.add_argument(
        "--controlled",
        type=Path,
        default=Path("outputs/controlled_seed7/metrics.csv"),
    )
    parser.add_argument(
        "--telecomts",
        type=Path,
        default=Path("outputs/telecomts_seed17/metrics.csv"),
    )
    parser.add_argument(
        "--multiseed",
        type=Path,
        default=Path("outputs/telecomts_multiseed/all_seeds.csv"),
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data/telecomts_800_even.jsonl.gz"),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=5e-4,
        help="Absolute tolerance; 5e-4 guarantees matching paper rounding.",
    )
    return parser.parse_args()


def _compare(
    actual_path: Path,
    expected_path: Path,
    keys: list[str],
    atol: float,
) -> None:
    if not actual_path.exists():
        raise FileNotFoundError(f"Missing generated artifact: {actual_path}")
    actual = pd.read_csv(actual_path).set_index(keys).sort_index()
    expected = pd.read_csv(expected_path).set_index(keys).sort_index()
    if not actual.index.equals(expected.index):
        raise AssertionError(
            f"Row keys differ for {actual_path}:\n"
            f"actual={list(actual.index)}\nexpected={list(expected.index)}"
        )
    for metric in METRIC_COLUMNS:
        delta = np.abs(
            actual[metric].to_numpy() - expected[metric].to_numpy()
        )
        if not np.all(delta <= atol):
            worst = int(np.argmax(delta))
            key = actual.index[worst]
            raise AssertionError(
                f"{actual_path}: {metric} differs at {key}: "
                f"actual={actual[metric].iloc[worst]:.10f}, "
                f"expected={expected[metric].iloc[worst]:.10f}, "
                f"|delta|={delta[worst]:.3g} > {atol}"
            )
    print(f"PASS {actual_path} (atol={atol:g})")


def main() -> None:
    args = parse_args()
    snapshot_hash = sha256_file(args.snapshot)
    if snapshot_hash != PAPER_CACHE_SHA256:
        raise AssertionError(
            f"Snapshot checksum differs: {snapshot_hash} "
            f"(expected {PAPER_CACHE_SHA256})"
        )
    print(f"PASS {args.snapshot} (sha256={snapshot_hash})")

    _compare(
        args.controlled,
        args.expected_dir / "controlled_seed7.csv",
        ["model"],
        args.atol,
    )
    _compare(
        args.telecomts,
        args.expected_dir / "telecomts_seed17.csv",
        ["model"],
        args.atol,
    )
    _compare(
        args.multiseed,
        args.expected_dir / "telecomts_multiseed_raw.csv",
        ["seed", "model"],
        args.atol,
    )
    print("All generated metrics reproduce the paper tables.")


if __name__ == "__main__":
    main()
