import unittest

import numpy as np

from benchmark.models import ridge_fit, ridge_predict


def primal_solution(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    penalty = np.eye(x_aug.shape[1])
    penalty[-1, -1] = 0.0
    return np.linalg.solve(
        x_aug.T @ x_aug + alpha * penalty,
        x_aug.T @ y,
    )


class RidgeTests(unittest.TestCase):
    def check_shape(self, n: int, d: int) -> None:
        rng = np.random.default_rng(9)
        x = rng.normal(size=(n, d))
        y = rng.normal(size=(n, 3))
        alpha = 2.5
        expected = primal_solution(x, y, alpha)
        actual = ridge_fit(x, y, alpha)
        np.testing.assert_allclose(
            ridge_predict(x, actual),
            ridge_predict(x, expected),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_primal_branch(self) -> None:
        self.check_shape(n=20, d=5)

    def test_dual_branch(self) -> None:
        self.check_shape(n=6, d=12)


if __name__ == "__main__":
    unittest.main()
