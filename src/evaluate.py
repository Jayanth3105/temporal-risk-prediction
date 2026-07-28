"""
=========================================================
Project : Temporal Risk Prediction for Autonomous Driving
Author  : Jayanth K
Version : 1.0.0
=========================================================

Evaluation Module

This module evaluates trajectory prediction performance.

Implemented Metrics

    • Average Displacement Error (ADE)
    • Final Displacement Error (FDE)
    • Root Mean Square Error (RMSE)

These metrics are commonly used in trajectory prediction
research.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


# =========================================================
# Validation
# =========================================================

def validate_shapes(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> None:
    """
    Validate prediction and ground-truth shapes.
    """

    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "Prediction and ground truth must have the same shape.\n"
            f"Prediction : {prediction.shape}\n"
            f"Ground Truth : {ground_truth.shape}"
        )


# =========================================================
# Average Displacement Error
# =========================================================

def compute_ade(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """
    Compute Average Displacement Error (ADE).

    Supports:
        (F,2)
        (B,F,2)
    """

    validate_shapes(prediction, ground_truth)

    error = torch.norm(
        prediction - ground_truth,
        dim=-1,
    )

    return error.mean().item()


# =========================================================
# Final Displacement Error
# =========================================================

def compute_fde(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """
    Compute Final Displacement Error (FDE).

    Supports:
        (F,2)
        (B,F,2)
    """

    validate_shapes(prediction, ground_truth)

    error = torch.norm(
        prediction[..., -1, :] -
        ground_truth[..., -1, :],
        dim=-1,
    )

    return error.mean().item()


# =========================================================
# Root Mean Square Error
# =========================================================

def compute_rmse(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """
    Compute Root Mean Square Error.
    """

    validate_shapes(prediction, ground_truth)

    mse = torch.mean(
        (prediction - ground_truth) ** 2
    )

    return torch.sqrt(mse).item()


# =========================================================
# Evaluate One Sample
# =========================================================

def evaluate_sample(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate one trajectory.
    """

    return {

        "ADE": compute_ade(
            prediction,
            ground_truth,
        ),

        "FDE": compute_fde(
            prediction,
            ground_truth,
        ),

        "RMSE": compute_rmse(
            prediction,
            ground_truth,
        ),
    }


# =========================================================
# Evaluate Dataset
# =========================================================

@torch.no_grad()
def evaluate_dataset(
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor],
) -> Dict[str, float]:
    """
    Evaluate an entire dataset.
    """

    if len(predictions) != len(ground_truths):
        raise ValueError(
            "Prediction list and Ground Truth list "
            "must have the same length."
        )

    ade_scores = []
    fde_scores = []
    rmse_scores = []

    for prediction, ground_truth in zip(
        predictions,
        ground_truths,
    ):

        metrics = evaluate_sample(
            prediction,
            ground_truth,
        )

        ade_scores.append(metrics["ADE"])
        fde_scores.append(metrics["FDE"])
        rmse_scores.append(metrics["RMSE"])

    return {

        "Samples": len(predictions),

        "ADE": float(np.mean(ade_scores)),

        "FDE": float(np.mean(fde_scores)),

        "RMSE": float(np.mean(rmse_scores)),
    }


# =========================================================
# Print Results
# =========================================================

def print_results(
    results: Dict[str, float],
) -> None:
    """
    Print evaluation summary.
    """

    print("=" * 45)
    print("Trajectory Prediction Evaluation")
    print("=" * 45)

    print(f"Samples : {results['Samples']}")
    print(f"ADE     : {results['ADE']:.4f} m")
    print(f"FDE     : {results['FDE']:.4f} m")
    print(f"RMSE    : {results['RMSE']:.4f} m")

    print("=" * 45)


# =========================================================
# Example
# =========================================================

if __name__ == "__main__":

    ground_truth = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
    ])

    prediction = ground_truth + (
        torch.randn_like(ground_truth) * 0.20
    )

    metrics = evaluate_sample(
        prediction,
        ground_truth,
    )

    print(metrics)
