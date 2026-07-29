import unittest

import numpy as np

from benchmark.dataset import (
    FEATURES,
    NODES,
    build_5gc_graph,
    generate_synthetic_5gc,
)
from benchmark.experiments import random_split


class DatasetTests(unittest.TestCase):
    def test_graph_planes_are_row_normalized(self) -> None:
        graph = build_5gc_graph()
        self.assertEqual(
            set(graph),
            {"control", "user", "slice", "all"},
        )
        for adjacency in graph.values():
            self.assertEqual(adjacency.shape, (len(NODES), len(NODES)))
            np.testing.assert_allclose(adjacency.sum(axis=1), 1.0)

    def test_controlled_generator_is_deterministic(self) -> None:
        first = generate_synthetic_5gc(
            timesteps=220,
            window=8,
            horizon=2,
            seed=5,
        )
        second = generate_synthetic_5gc(
            timesteps=220,
            window=8,
            horizon=2,
            seed=5,
        )
        self.assertEqual(
            first.x.shape,
            (211, 8, len(NODES), len(FEATURES)),
        )
        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(first.y_latency, second.y_latency)
        np.testing.assert_array_equal(first.y_violation, second.y_violation)

    def test_seeded_row_split_is_disjoint(self) -> None:
        train, validation, test = random_split(100, seed=17)
        combined = np.concatenate([train, validation, test])
        self.assertEqual(len(train), 65)
        self.assertEqual(len(validation), 15)
        self.assertEqual(len(test), 20)
        self.assertEqual(len(np.unique(combined)), 100)


if __name__ == "__main__":
    unittest.main()
