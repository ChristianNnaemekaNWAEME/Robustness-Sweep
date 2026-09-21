"""
metrics.py
==========

The three evaluation metrics used in the published thesis: Root Mean
Square Error, Mean Absolute Deviation, and Median Absolute Error.
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mad(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_true - y_pred)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAD": mad(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
    }
