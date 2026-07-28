"""
=========================================================
Project : Temporal Risk Prediction for Autonomous Driving
Author : Jayanth K
Version : 1.0.0
=========================================================

FrameData

Represents one frame flowing through the perception
pipeline.

Each processing stage (detection, tracking, prediction,
risk estimation, visualization) updates this object
instead of passing multiple variables between functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

import config
from tracked_object import TrackedObject

# =========================================================
# Detection
# =========================================================

@dataclass(slots=True)
class Detection:
    """
    Represents a single object detection produced by
    the detector before tracking.
    """

    class_id: int

    class_name: str

    confidence: float

    bbox: tuple[int, int, int, int]

    center: tuple[float, float]


# =========================================================
# Frame Data
# =========================================================

@dataclass(slots=True)
class FrameData:
    """
    Container representing one frame moving through the
    complete perception pipeline.
    """

    # -----------------------------------------------------
    # Frame Information
    # -----------------------------------------------------

    frame_number: int

    image: np.ndarray

    # Argoverse 2 uses nanosecond timestamps
    timestamp_ns: int = 0

    camera_name: str = config.CAMERA_NAME

    # -----------------------------------------------------
    # Perception
    # -----------------------------------------------------

    detections: List[Detection] = field(default_factory=list)

    tracked_objects: List[TrackedObject] = field(default_factory=list)

    # Predictions written by the temporal model
    predicted_objects: List[TrackedObject] = field(default_factory=list)

    # -----------------------------------------------------
    # Utility Methods
    # -----------------------------------------------------

    def clear_detections(self) -> None:
        """Remove all detections."""
        self.detections.clear()

    def clear_tracks(self) -> None:
        """Remove all tracked objects."""
        self.tracked_objects.clear()

    def clear_predictions(self) -> None:
        """Remove all predicted objects."""
        self.predicted_objects.clear()

    def add_detection(self, detection: Detection) -> None:
        """Add a detection."""
        self.detections.append(detection)

    def add_track(self, track: TrackedObject) -> None:
        """Add a tracked object."""
        self.tracked_objects.append(track)

    def add_prediction(self, track: TrackedObject) -> None:
        """Add a predicted tracked object."""
        self.predicted_objects.append(track)

    @property
    def detection_count(self) -> int:
        """Return number of detections."""
        return len(self.detections)

    @property
    def track_count(self) -> int:
        """Return number of tracked objects."""
        return len(self.tracked_objects)

    @property
    def prediction_count(self) -> int:
        """Return number of predicted objects."""
        return len(self.predicted_objects)

    def reset(self) -> None:
        """
        Clear all perception results while keeping
        frame metadata.
        """
        self.clear_detections()
        self.clear_tracks()
        self.clear_predictions()

    def copy_metadata(self) -> "FrameData":
        """
        Create a new FrameData object with the same
        frame metadata but without detections/tracks/predictions.
        """
        return FrameData(
            frame_number=self.frame_number,
            image=self.image,
            timestamp_ns=self.timestamp_ns,
            camera_name=self.camera_name,
        )

    def __len__(self) -> int:
        """
        Returns the number of tracked objects if
        available; otherwise returns detections.
        """
        if self.track_count > 0:
            return self.track_count

        return self.detection_count

    def __str__(self) -> str:
        return (
            "FrameData("
            f"frame={self.frame_number}, "
            f"detections={self.detection_count}, "
            f"tracks={self.track_count}, "
            f"predictions={self.prediction_count})"
        )
