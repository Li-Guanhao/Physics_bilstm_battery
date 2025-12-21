import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error

def mean_relative_error(true, pred, epsilon=1e-8):
    return np.mean(np.abs((true - pred) / (true + epsilon)))

def evaluate_metrics(true_flat, pred_flat, columns):
    for i, col in enumerate(columns):
        rmse = root_mean_squared_error(true_flat[:, i], pred_flat[:, i])
        mre = mean_relative_error(true_flat[:, i], pred_flat[:, i])
        print(f"Test RMSE {col:13s}: {rmse:.4f}")
        print(f"Test MRE {col:13s}: {mre:.4f}")