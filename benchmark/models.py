from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


class Standardizer:
    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = x.mean(axis=0, keepdims=True)
        self.std = x.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Fit ridge regression with an unregularized intercept.

    TelecomTS has many more flattened features than training records.  The
    centered dual form is algebraically equivalent to the primal normal
    equations but avoids materializing a 15k-by-15k Gram matrix.
    """
    x_mean = x.mean(axis=0, keepdims=True)
    y_2d = y if y.ndim == 2 else y[:, None]
    y_mean = y_2d.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    y_centered = y_2d - y_mean

    if x.shape[1] > x.shape[0]:
        dual = np.linalg.solve(
            x_centered @ x_centered.T + alpha * np.eye(len(x)),
            y_centered,
        )
        coef = x_centered.T @ dual
    else:
        coef = np.linalg.solve(
            x_centered.T @ x_centered + alpha * np.eye(x.shape[1]),
            x_centered.T @ y_centered,
        )
    intercept = y_mean - x_mean @ coef
    weights = np.concatenate([coef, intercept], axis=0)
    return weights[:, 0] if y.ndim == 1 else weights


def ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    return x_aug @ w


@dataclass
class Prediction:
    latency: np.ndarray
    violation: np.ndarray


class BaseModel:
    name = "base"

    def fit(self, x: np.ndarray, y_latency: np.ndarray, y_violation: np.ndarray) -> "BaseModel":
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> Prediction:
        raise NotImplementedError


class PersistenceBaseline(BaseModel):
    name = "persistence"

    def __init__(self, slice_indices=(8, 9, 10), latency_feature=5, sla=(45.0, 12.0, 80.0)):
        self.slice_indices = slice_indices
        self.latency_feature = latency_feature
        self.sla = np.asarray(sla)

    def fit(self, x: np.ndarray, y_latency: np.ndarray, y_violation: np.ndarray) -> "PersistenceBaseline":
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        lat = x[:, -1, self.slice_indices, self.latency_feature]
        score = sigmoid((lat - self.sla) / np.maximum(self.sla * 0.08, 1.0))
        return Prediction(latency=lat, violation=score)


class RidgeBaseline(BaseModel):
    name = "ridge_flat"

    def __init__(self, alpha: float = 20.0):
        self.alpha = alpha
        self.scaler = Standardizer()

    def _features(self, x: np.ndarray) -> np.ndarray:
        flat = x.reshape(len(x), -1)
        stats = np.concatenate(
            [
                x[:, -1].reshape(len(x), -1),
                x.mean(axis=1).reshape(len(x), -1),
                x.std(axis=1).reshape(len(x), -1),
            ],
            axis=1,
        )
        return np.concatenate([flat, stats], axis=1)

    def fit(self, x: np.ndarray, y_latency: np.ndarray, y_violation: np.ndarray) -> "RidgeBaseline":
        feats = self.scaler.fit_transform(self._features(x))
        self.w_latency = ridge_fit(feats, y_latency, self.alpha)
        self.w_violation = ridge_fit(feats, y_violation, self.alpha)
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        feats = self.scaler.transform(self._features(x))
        lat = ridge_predict(feats, self.w_latency)
        vio = sigmoid(ridge_predict(feats, self.w_violation))
        return Prediction(latency=lat, violation=vio)


class TemporalMLP(BaseModel):
    name = "temporal_mlp"

    def __init__(self, hidden: int = 96, lr: float = 0.018, epochs: int = 170, seed: int = 3, alpha: float = 2e-4):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.alpha = alpha
        self.scaler = Standardizer()

    def _features(self, x: np.ndarray) -> np.ndarray:
        recent = x[:, -4:].reshape(len(x), -1)
        trend = (x[:, -1] - x[:, 0]).reshape(len(x), -1)
        return np.concatenate([recent, x.mean(axis=1).reshape(len(x), -1), trend], axis=1)

    def fit(self, x: np.ndarray, y_latency: np.ndarray, y_violation: np.ndarray) -> "TemporalMLP":
        rng = np.random.default_rng(self.seed)
        feats = self.scaler.fit_transform(self._features(x))
        y = np.concatenate([y_latency / 100.0, y_violation], axis=1)
        n, d = feats.shape
        out = y.shape[1]
        self.w1 = rng.normal(0, np.sqrt(2 / d), size=(d, self.hidden))
        self.b1 = np.zeros((1, self.hidden))
        self.w2 = rng.normal(0, np.sqrt(2 / self.hidden), size=(self.hidden, out))
        self.b2 = np.zeros((1, out))
        batch = min(128, n)
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch):
                ix = order[start : start + batch]
                xb, yb = feats[ix], y[ix]
                h = np.tanh(xb @ self.w1 + self.b1)
                pred = h @ self.w2 + self.b2
                pred[:, 3:] = sigmoid(pred[:, 3:])
                grad = (pred - yb) / len(ix)
                grad[:, 3:] *= pred[:, 3:] * (1.0 - pred[:, 3:])
                gw2 = h.T @ grad + self.alpha * self.w2
                gb2 = grad.sum(axis=0, keepdims=True)
                gh = grad @ self.w2.T * (1.0 - h**2)
                gw1 = xb.T @ gh + self.alpha * self.w1
                gb1 = gh.sum(axis=0, keepdims=True)
                self.w2 -= self.lr * gw2
                self.b2 -= self.lr * gb2
                self.w1 -= self.lr * gw1
                self.b1 -= self.lr * gb1
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        feats = self.scaler.transform(self._features(x))
        h = np.tanh(feats @ self.w1 + self.b1)
        out = h @ self.w2 + self.b2
        return Prediction(latency=np.maximum(out[:, :3] * 100.0, 0.0), violation=sigmoid(out[:, 3:]))


class STwinGNNLite(BaseModel):
    name = "s_twingnn_lite"

    def __init__(
        self,
        graph: Dict[str, np.ndarray],
        alpha: float = 4.0,
        diffusion_steps: int = 2,
        random_dim: int = 384,
        temporal_decay: float = 0.82,
        queue_weight: float = 1.0,
        seed: int = 11,
        slice_indices=(8, 9, 10),
        sla=(45.0, 12.0, 80.0),
        residual_scale: float = 0.65,
    ):
        self.graph = graph
        self.alpha = alpha
        self.diffusion_steps = diffusion_steps
        self.random_dim = random_dim
        self.temporal_decay = temporal_decay
        self.queue_weight = queue_weight
        self.seed = seed
        self.slice_indices = slice_indices
        self.sla = np.asarray(sla)
        self.residual_scale = residual_scale
        self.scaler = Standardizer()

    def _diffuse(self, x_t: np.ndarray, adj: np.ndarray) -> np.ndarray:
        h = x_t
        outs = [h]
        for _ in range(self.diffusion_steps):
            h = np.einsum("ij,bjf->bif", adj, h)
            outs.append(h)
        return np.concatenate(outs, axis=-1)

    def _queue_features(self, x: np.ndarray) -> np.ndarray:
        arrival = x[..., 8]
        service = np.maximum(x[..., 9], arrival + 1e-3)
        util = np.clip(arrival / service, 0.0, 0.995)
        delay = self.queue_weight / np.maximum(service - arrival, 1.0)
        return np.stack([util, delay, util**2], axis=-1)

    def _features(self, x: np.ndarray) -> np.ndarray:
        weights = self.temporal_decay ** np.arange(x.shape[1] - 1, -1, -1)
        weights = weights / weights.sum()
        x_att = np.einsum("t,btnd->bnd", weights, x)
        x_last = x[:, -1]
        x_trend = x[:, -1] - x[:, 0]

        control = self._diffuse(x_att, self.graph["control"])
        user = self._diffuse(x_att, self.graph["user"])
        slice_g = self._diffuse(x_att, self.graph["slice"])
        all_g = self._diffuse(x_last, self.graph["all"])
        queue = self._queue_features(x[:, -4:]).mean(axis=1)
        recent_raw = x[:, -4:].reshape(len(x), -1)
        last_raw = x[:, -1].reshape(len(x), -1)
        mean_raw = x.mean(axis=1).reshape(len(x), -1)
        std_raw = x.std(axis=1).reshape(len(x), -1)
        slice_last = x[:, -1, self.slice_indices, :].reshape(len(x), -1)
        slice_trend = (x[:, -1, self.slice_indices, :] - x[:, 0, self.slice_indices, :]).reshape(len(x), -1)

        cross_plane = control * user
        trend_slice = self._diffuse(x_trend, self.graph["slice"])
        feats = np.concatenate(
            [
                recent_raw,
                last_raw,
                mean_raw,
                std_raw,
                slice_last,
                slice_trend,
                control.reshape(len(x), -1),
                user.reshape(len(x), -1),
                slice_g.reshape(len(x), -1),
                all_g.reshape(len(x), -1),
                cross_plane.reshape(len(x), -1),
                trend_slice.reshape(len(x), -1),
                queue.reshape(len(x), -1),
            ],
            axis=1,
        )
        return feats

    def fit(self, x: np.ndarray, y_latency: np.ndarray, y_violation: np.ndarray) -> "STwinGNNLite":
        rng = np.random.default_rng(self.seed)
        feats = self.scaler.fit_transform(self._features(x))
        if self.random_dim > 0:
            proj_scale = 1.0 / np.sqrt(feats.shape[1])
            self.random_w = rng.normal(0, proj_scale, size=(feats.shape[1], self.random_dim))
            self.random_b = rng.uniform(0, 2 * np.pi, size=(1, self.random_dim))
            z = np.concatenate([feats, np.cos(feats @ self.random_w + self.random_b)], axis=1)
        else:
            self.random_w = None
            self.random_b = None
            z = feats
        base_latency = x[:, -1, self.slice_indices, 5]
        residual = y_latency - base_latency
        self.residual_clip = np.quantile(np.abs(residual), 0.90, axis=0) + 1e-6
        self.w_latency = ridge_fit(z, residual, self.alpha)
        self.w_violation = ridge_fit(z, y_violation, self.alpha)
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        feats = self.scaler.transform(self._features(x))
        if self.random_w is None:
            z = feats
        else:
            z = np.concatenate([feats, np.cos(feats @ self.random_w + self.random_b)], axis=1)
        base_latency = x[:, -1, self.slice_indices, 5]
        residual = np.clip(ridge_predict(z, self.w_latency), -self.residual_clip, self.residual_clip)
        lat = np.maximum(base_latency + self.residual_scale * residual, 0.0)
        learned_vio = sigmoid(ridge_predict(z, self.w_violation))
        sla_vio = sigmoid((lat - self.sla) / np.maximum(self.sla * 0.07, 0.8))
        vio = 0.35 * learned_vio + 0.65 * sla_vio
        return Prediction(latency=lat, violation=vio)


def tune_stwingnn(
    graph: Dict[str, np.ndarray],
    x_train: np.ndarray,
    y_lat_train: np.ndarray,
    y_vio_train: np.ndarray,
    x_val: np.ndarray,
    y_lat_val: np.ndarray,
    y_vio_val: np.ndarray,
    metric_fn,
) -> Tuple[STwinGNNLite, dict]:
    best_model: Optional[STwinGNNLite] = None
    best_score = float("inf")
    best_metrics = {}
    grid = [
        {"alpha": 12.0, "diffusion_steps": 1, "random_dim": 0, "temporal_decay": 0.78, "queue_weight": 0.8, "residual_scale": 0.35},
        {"alpha": 25.0, "diffusion_steps": 2, "random_dim": 0, "temporal_decay": 0.82, "queue_weight": 1.0, "residual_scale": 0.50},
        {"alpha": 60.0, "diffusion_steps": 2, "random_dim": 0, "temporal_decay": 0.88, "queue_weight": 1.2, "residual_scale": 0.65},
        {"alpha": 90.0, "diffusion_steps": 2, "random_dim": 0, "temporal_decay": 0.90, "queue_weight": 1.2, "residual_scale": 0.35},
        {"alpha": 8.0, "diffusion_steps": 2, "random_dim": 128, "temporal_decay": 0.86, "queue_weight": 1.0, "residual_scale": 0.50},
        {"alpha": 30.0, "diffusion_steps": 3, "random_dim": 128, "temporal_decay": 0.90, "queue_weight": 1.4, "residual_scale": 0.35},
    ]
    for params in grid:
        model = STwinGNNLite(graph=graph, **params).fit(x_train, y_lat_train, y_vio_train)
        pred = model.predict(x_val)
        metrics = metric_fn(y_lat_val, pred.latency, y_vio_val, pred.violation)
        score = metrics["latency_mae"] - 6.0 * metrics["violation_f1"]
        if score < best_score:
            best_score = score
            best_model = model
            best_metrics = {**metrics, **params}
    assert best_model is not None
    return best_model, best_metrics


class HybridDigitalTwin(BaseModel):
    name = "hybrid_dt"

    def __init__(self, graph: Dict[str, np.ndarray], ridge_alpha: float = 35.0, mlp_seed: int = 17):
        self.graph_head = STwinGNNLite(
            graph=graph,
            alpha=ridge_alpha,
            diffusion_steps=2,
            random_dim=0,
            temporal_decay=0.88,
            queue_weight=1.2,
            residual_scale=0.65,
        )
        self.latency_fallback = RidgeBaseline(alpha=ridge_alpha)
        self.violation_head = TemporalMLP(hidden=96, lr=0.012, epochs=180, seed=mlp_seed)

    def fit(self, x: np.ndarray, y_latency: np.ndarray, y_violation: np.ndarray) -> "HybridDigitalTwin":
        self.graph_head.fit(x, y_latency, y_violation)
        self.latency_fallback.fit(x, y_latency, y_violation)
        self.violation_head.fit(x, y_latency, y_violation)
        return self

    def predict(self, x: np.ndarray) -> Prediction:
        graph_pred = self.graph_head.predict(x)
        ridge_pred = self.latency_fallback.predict(x)
        mlp_pred = self.violation_head.predict(x)
        latency = 0.25 * graph_pred.latency + 0.75 * ridge_pred.latency
        violation = 0.15 * graph_pred.violation + 0.85 * mlp_pred.violation
        return Prediction(latency=latency, violation=violation)
