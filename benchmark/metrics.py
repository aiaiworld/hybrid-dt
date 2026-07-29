from __future__ import annotations

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-6)) * 100.0
    return {"latency_mae": float(mae), "latency_rmse": float(rmse), "latency_mape": float(mape)}


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true_flat = y_true.reshape(-1).astype(int)
    y_pred_flat = (y_score.reshape(-1) >= threshold).astype(int)
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "violation_accuracy": float(accuracy),
        "violation_precision": float(precision),
        "violation_recall": float(recall),
        "violation_f1": float(f1),
    }


def combined_metrics(y_lat: np.ndarray, y_lat_pred: np.ndarray, y_vio: np.ndarray, y_vio_score: np.ndarray) -> dict:
    out = regression_metrics(y_lat, y_lat_pred)
    out.update(classification_metrics(y_vio, y_vio_score))
    return out
