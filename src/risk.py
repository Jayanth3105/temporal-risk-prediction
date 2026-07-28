"""
=========================================================
Project : Temporal Risk Prediction for Autonomous Driving
Author  : Jayanth K
Version : 1.0.0
=========================================================

Risk Assessment Module

This module estimates the collision risk of tracked
objects using predicted future trajectories.

Pipeline

Predicted Trajectory
        ↓
Distance
        ↓
Relative Speed
        ↓
Time-To-Collision
        ↓
Risk Score
        ↓
LOW / MEDIUM / HIGH
"""

from __future__ import annotations

import math
from typing import Tuple

import config
from tracked_object import TrackedObject


# =========================================================
# Distance
# =========================================================

def calculate_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
) -> float:
    """
    Calculate Euclidean distance.
    """
    return math.dist(point1, point2)


# =========================================================
# Relative Speed
# =========================================================

def calculate_speed(
    velocity: Tuple[float, float],
) -> float:
    """
    Calculate object speed.
    """
    vx, vy = velocity
    return math.sqrt(vx ** 2 + vy ** 2)


# =========================================================
# Time To Collision
# =========================================================

def calculate_ttc(
    distance: float,
    speed: float,
) -> float:
    """
    Calculate Time-To-Collision.

    TTC = distance / speed
    """
    if speed <= 0:
        return float("inf")

    return distance / speed


# =========================================================
# Risk Score
# =========================================================

def calculate_risk_score(
    distance: float,
    speed: float,
    ttc: float,
) -> float:
    """
    Calculate normalized risk score.

    Combines:
        - distance_score
        - speed_score
        - ttc_score

    with weights:
        0.4 * distance_score
        0.3 * speed_score
        0.3 * ttc_score
    """

    # Normalize distance: closer → higher risk
    distance_score = max(
        0.0,
        1.0 - distance / 30.0,
    )

    # Normalize speed: faster → higher risk
    speed_score = min(
        speed / 15.0,
        1.0,
    )

    # Normalize TTC: smaller TTC → higher risk
    if math.isinf(ttc):
        ttc_score = 0.0
    else:
        ttc_score = max(
            0.0,
            1.0 - ttc / 5.0,
        )

    risk = (
        0.4 * distance_score +
        0.3 * speed_score +
        0.3 * ttc_score
    )

    return max(
        0.0,
        min(risk, 1.0),
    )


# =========================================================
# Risk Classification
# =========================================================

def classify_risk(
    score: float,
) -> str:
    """
    Convert risk score into category.
    """
    if score >= config.HIGH_RISK_THRESHOLD:
        return "HIGH"

    if score >= config.MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"

    return "LOW"


# =========================================================
# Evaluate Risk
# =========================================================

def evaluate_risk(
    tracked_object: TrackedObject,
    ego_position: Tuple[float, float] = (0.0, 0.0),
) -> TrackedObject:
    """
    Estimate collision risk for one tracked object.

    Uses:
        - current world_position
        - current velocity
        - TTC and normalized risk score

    and writes back:
        tracked_object.risk_score
        tracked_object.risk_level
        tracked_object.ttc
    """
    distance = calculate_distance(
        tracked_object.world_position,
        ego_position,
    )

    speed = calculate_speed(
        tracked_object.velocity,
    )

    ttc = calculate_ttc(
        distance,
        speed,
    )

    score = calculate_risk_score(
        distance,
        speed,
        ttc,
    )

    level = classify_risk(score)

    tracked_object.update_risk(
        risk_score=score,
        risk_level=level,
        ttc=ttc,
    )

    return tracked_object


# =========================================================
# Example
# =========================================================

if __name__ == "__main__":

    obj = TrackedObject(
        track_id=1,
        track_uuid="1",
        class_name="PEDESTRIAN",
        confidence=0.95,
        bbox=(100, 100, 150, 250),
        center=(125.0, 175.0),
    )

    # Set world position directly
    obj.world_position = (8.0, 2.0)

    # Example velocity
    obj.velocity = (1.5, 0.2)

    obj = evaluate_risk(obj)

    print("Risk Score :", obj.risk_score)
    print("Risk Level :", obj.risk_level)
    print("TTC        :", obj.ttc)
