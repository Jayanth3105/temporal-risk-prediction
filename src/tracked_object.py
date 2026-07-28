"""
=========================================================
Project : Temporal Risk Prediction for Autonomous Driving
Author  : Jayanth K
Version : 0.2.0
=========================================================

Tracked Object

Universal tracked-object representation used across:

    - detect.py              (YOLO detections)
    - track.py               (ByteTrack associations)
    - trajectory.py          (TrajectoryManager)
    - predict.py             (TemporalPredictor)
    - risk.py                (risk estimation)
    - evaluate.py            (metrics)
    - results_visualizer.py  (visualization)

This class is intentionally focused on temporal motion
prediction and risk-aware autonomous driving. It is NOT a
generic autonomous-driving object with full kinematic state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import config


# =========================================================
# Enumerations
# =========================================================

class MotionState(Enum):
    """
    High-level motion state of a tracked object.
    """

    UNKNOWN = auto()
    STATIC = auto()
    MOVING = auto()
    AGGRESSIVE = auto()


# =========================================================
# Tracked Object
# =========================================================

@dataclass
class TrackedObject:
    """
    Universal representation of a dynamic road user.

    Identity:
        - track_id        : stable integer ID from tracker
        - track_uuid      : optional string UUID
        - class_name      : semantic class label
        - confidence      : detector confidence

    Spatial:
        - bbox            : (x1, y1, x2, y2) in image pixels
        - center          : (cx, cy) in image pixels
        - world_position  : (x, y) in meters (e.g., Argoverse tx_m, ty_m)

    Temporal:
        - timestamp_ns        : annotation timestamp (ns)
        - image_timestamp_ns  : matched camera frame timestamp (ns)
        - velocity            : estimated (vx, vy)
        - acceleration        : not currently used, reserved
        - motion_state        : STATIC / MOVING / AGGRESSIVE / UNKNOWN

    Trajectories:
        - history             : past positions (for prediction)
        - future_ground_truth : GT future positions (training/eval)
        - future_prediction   : model future positions
        - feature_vector      : CNN features (optional)

    Risk & evaluation:
        - risk_score          : continuous [0, 1]
        - risk_level          : "LOW" / "MEDIUM" / "HIGH"
        - ttc                 : time-to-collision (seconds)
        - ade, fde            : ADE / FDE metrics

    Lifecycle:
        - active              : whether currently active
    """

    # -----------------------------------------------------
    # Identity
    # -----------------------------------------------------

    track_id: int
    track_uuid: str
    class_name: str
    confidence: float = 0.0

    # -----------------------------------------------------
    # Spatial representation
    # -----------------------------------------------------

    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    center: Tuple[float, float] = (0.0, 0.0)
    world_position: Tuple[float, float] = (0.0, 0.0)
    anchor_world_position: Tuple[float, float] = (0.0, 0.0)

    # -----------------------------------------------------
    # Temporal state
    # -----------------------------------------------------

    timestamp_ns: int = 0
    image_timestamp_ns: int = 0  # matched camera frame timestamp

    velocity: Tuple[float, float] = (0.0, 0.0)
    acceleration: Tuple[float, float] = (0.0, 0.0)
    motion_state: MotionState = MotionState.UNKNOWN

    # -----------------------------------------------------
    # Trajectory history & predictions
    # -----------------------------------------------------

    history: List[Tuple[float, float]] = field(default_factory=list)
    future_ground_truth: List[Tuple[float, float]] = field(default_factory=list)
    future_prediction: List[Tuple[float, float]] = field(default_factory=list)

    feature_vector: Optional[List[float]] = None

    # -----------------------------------------------------
    # Risk estimation
    # -----------------------------------------------------

    risk_score: float = 0.0
    risk_level: str = "LOW"
    ttc: float = float("inf")

    # -----------------------------------------------------
    # Evaluation metrics
    # -----------------------------------------------------

    ade: float = 0.0
    fde: float = 0.0

    # -----------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------

    active: bool = True

    # =====================================================
    # Construction Helpers
    # =====================================================

    @classmethod
    def from_detection(
        cls,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        class_name: str,
        confidence: float,
        timestamp_ns: int,
    ) -> TrackedObject:
        """
        Construct a TrackedObject from a detector output.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return cls(
            track_id=track_id,
            track_uuid=str(track_id),
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            center=(cx, cy),
            timestamp_ns=timestamp_ns,
            image_timestamp_ns=timestamp_ns,
        )

    # =====================================================
    # Lifecycle
    # =====================================================

    def activate(self) -> None:
        """
        Mark object as active.
        """
        self.active = True

    def deactivate(self) -> None:
        """
        Mark object as inactive.
        """
        self.active = False

    def reset(self) -> None:
        """
        Reset temporal, risk, and evaluation state.

        Identity (track_id, track_uuid, class_name) and
        current bbox/center are preserved.
        """
        # Trajectory history and predictions
        self.clear_history()
        self.clear_prediction()

        # Ground truth and features
        self.future_ground_truth.clear()
        self.feature_vector = None

        # Reset anchor metadata
        self.anchor_world_position = (0.0, 0.0)

        # Risk-related fields
        self.risk_score = 0.0
        self.risk_level = "LOW"
        self.ttc = float("inf")

        # Evaluation metrics
        self.ade = 0.0
        self.fde = 0.0

        # Kinematics
        self.velocity = (0.0, 0.0)
        self.acceleration = (0.0, 0.0)
        self.motion_state = MotionState.UNKNOWN

    # =====================================================
    # Trajectory Management
    # =====================================================

    def add_history_point(self, x: float, y: float) -> None:
        """
        Append a new position to the history trajectory.
        """
        self.history.append((x, y))

    def add_history(self, point: Tuple[float, float]) -> None:
        """
        Backwards-compatible wrapper used by TrajectoryManager.
        """
        x, y = point
        self.add_history_point(x, y)

    def clear_history(self) -> None:
        """
        Clear trajectory history.
        """
        self.history.clear()

    def get_history_length(self) -> int:
        """
        Return the number of history points.
        """
        return len(self.history)

    def has_enough_history(self, required: int) -> bool:
        """
        Return True if there are enough history points
        for prediction (used by TrajectoryManager).
        """
        return len(self.history) >= required

    def add_future_ground_truth(self, x: float, y: float) -> None:
        """
        Append a ground-truth future position.
        """
        self.future_ground_truth.append((x, y))

    def set_prediction(self, future_points: List[Tuple[float, float]]) -> None:
        """
        Set predicted future trajectory.
        """
        self.future_prediction = list(future_points)

    def clear_prediction(self) -> None:
        """
        Clear predicted future trajectory.
        """
        self.future_prediction.clear()

    # =====================================================
    # Kinematics
    # =====================================================

    def calculate_velocity(self) -> Tuple[float, float]:
        """
        Estimate velocity using the last two history positions.

        Currently uses simple difference:

            vx = x2 - x1
            vy = y2 - y1

        Later you can make this time-aware:

            dt = 1.0 / config.FRAME_RATE
            vx = (x2 - x1) / dt
            vy = (y2 - y1) / dt
        """
        if len(self.history) < 2:
            self.velocity = (0.0, 0.0)
            return self.velocity

        x1, y1 = self.history[-2]
        x2, y2 = self.history[-1]

        vx = x2 - x1
        vy = y2 - y1

        self.velocity = (vx, vy)
        return self.velocity

    def update_motion_state(self) -> None:
        """
        Update motion_state from velocity magnitude.

        Basic logic:
            - STATIC    : speed ≈ 0
            - MOVING    : speed > 0
            - AGGRESSIVE: speed above a heuristic threshold

        Thresholds are simple; they can be refined later.
        """
        vx, vy = self.velocity
        speed = (vx ** 2 + vy ** 2) ** 0.5

        # Small epsilon to avoid noise
        eps = 1e-3
        aggressive_threshold = 5.0  # heuristic units

        if speed < eps:
            self.motion_state = MotionState.STATIC
        elif speed > aggressive_threshold:
            self.motion_state = MotionState.AGGRESSIVE
        else:
            self.motion_state = MotionState.MOVING

    def update_frame(self, frame_index: int) -> None:
        """
        Attach current frame index to the object.

        TrajectoryManager uses this to track age and lifecycle.
        """
        setattr(self, "frame_index", int(frame_index))

    def update_bbox(self, bbox: Tuple[int, int, int, int]) -> None:
        """
        Update bounding box and recompute center.
        """
        self.bbox = bbox
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        self.center = (cx, cy)

    def update_center(self, center: Tuple[float, float]) -> None:
        """
        Update center only (e.g., if bbox is unchanged).
        """
        self.center = (float(center[0]), float(center[1]))

    def update_confidence(self, confidence: float) -> None:
        """
        Update detector confidence.
        """
        self.confidence = float(confidence)

    # =====================================================
    # Risk & Evaluation
    # =====================================================

    def update_risk(
        self,
        risk_score: float,
        risk_level: str,
        ttc: Optional[float] = None,
    ) -> None:
        """
        Update risk-related fields.

        risk_level is expected to be one of:
            "LOW", "MEDIUM", "HIGH"
        """
        self.risk_score = float(risk_score)
        self.risk_level = str(risk_level).upper()

        if ttc is not None:
            self.ttc = float(ttc)

    def update_metrics(
        self,
        ade: Optional[float] = None,
        fde: Optional[float] = None,
    ) -> None:
        """
        Update evaluation metrics (ADE/FDE).
        """
        if ade is not None:
            self.ade = float(ade)
        if fde is not None:
            self.fde = float(fde)

    # =====================================================
    # Utility
    # =====================================================

    def __repr__(self) -> str:
        return (
            f"TrackedObject("
            f"id={self.track_id}, "
            f"class={self.class_name}, "
            f"conf={self.confidence:.2f}, "
            f"risk={self.risk_level}({self.risk_score:.2f}), "
            f"history_len={len(self.history)}, "
            f"pred_len={len(self.future_prediction)})"
        )
