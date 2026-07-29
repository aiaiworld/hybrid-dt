import unittest
from pathlib import Path

import numpy as np

from benchmark.dataset import FEATURES, NODES
from benchmark.telecomts import (
    PAPER_CACHE_SHA256,
    load_telecomts_bundle,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "telecomts_800_even.jsonl.gz"


class SnapshotTests(unittest.TestCase):
    def test_snapshot_checksum_and_projection(self) -> None:
        self.assertEqual(sha256_file(SNAPSHOT), PAPER_CACHE_SHA256)
        bundle = load_telecomts_bundle(
            cache_path=SNAPSHOT,
            samples=5,
            input_len=96,
            allow_download=False,
            expected_sha256=PAPER_CACHE_SHA256,
        )
        self.assertEqual(bundle.x.shape, (5, 96, len(NODES), len(FEATURES)))
        self.assertEqual(bundle.y_latency.shape, (5, 3))
        self.assertEqual(bundle.y_violation.shape, (5, 3))
        self.assertTrue(np.isfinite(bundle.x).all())
        self.assertTrue(np.isfinite(bundle.y_latency).all())


if __name__ == "__main__":
    unittest.main()
