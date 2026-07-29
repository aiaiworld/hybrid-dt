import unittest

import numpy as np

from benchmark.metrics import classification_metrics, regression_metrics


class MetricTests(unittest.TestCase):
    def test_regression_metrics(self) -> None:
        truth = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = np.array([[2.0, 2.0], [1.0, 5.0]])
        result = regression_metrics(truth, pred)
        self.assertAlmostEqual(result["latency_mae"], 1.0)
        self.assertAlmostEqual(result["latency_rmse"], np.sqrt(1.5))

    def test_classification_metrics(self) -> None:
        truth = np.array([[1, 0], [1, 0]])
        score = np.array([[0.9, 0.6], [0.8, 0.1]])
        result = classification_metrics(truth, score)
        self.assertAlmostEqual(result["violation_accuracy"], 0.75)
        self.assertAlmostEqual(result["violation_precision"], 2.0 / 3.0)
        self.assertAlmostEqual(result["violation_recall"], 1.0)
        self.assertAlmostEqual(result["violation_f1"], 0.8)


if __name__ == "__main__":
    unittest.main()
