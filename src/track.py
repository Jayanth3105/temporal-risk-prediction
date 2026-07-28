"""
ByteTrack integration for temporal risk prediction.

This module converts project-native detections into ByteTrack inputs, updates
track association, synchronizes :class:`TrajectoryManager`, and stores active
tracked objects back into :class:`FrameData`.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import config
from tracked_object import TrackedObject
from trajectory import TrajectoryManager

try:
    from frame_data import Detection, FrameData
except ModuleNotFoundError:
    from src.frame_data import Detection, FrameData


@dataclass(frozen=True)
class _MatchedTrack:
    """Decoded active track returned by ByteTrack."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]


class _ByteTrackDetections:
    """Results-like detection container expected by Ultralytics ByteTrack."""

    def __init__(
        self,
        xywh: np.ndarray,
        confidence: np.ndarray,
        class_ids: np.ndarray,
    ) -> None:
        """Create an immutable ByteTrack detection view.

        Args:
            xywh: Detection boxes as center-x, center-y, width, height.
            confidence: Detection confidence scores.
            class_ids: Integer detection class identifiers.
        """
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(class_ids, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        """Return number of detections."""
        return int(self.conf.shape[0])

    def __getitem__(self, index: Any) -> "_ByteTrackDetections":
        """Return a sliced detection container."""
        return _ByteTrackDetections(
            xywh=self.xywh[index],
            confidence=self.conf[index],
            class_ids=self.cls[index],
        )


class Tracker:
    """ByteTrack wrapper for project-native perception data."""

    _BBOX_X_MIN = 0
    _BBOX_Y_MIN = 1
    _BBOX_X_MAX = 2
    _BBOX_Y_MAX = 3
    _OUTPUT_TRACK_ID = 4
    _OUTPUT_SCORE = 5
    _OUTPUT_CLASS_ID = 6
    _OUTPUT_DETECTION_INDEX = 7
    _OUTPUT_MIN_COLUMNS = 8
    _CENTER_DIVISOR = 2.0
    _DEFAULT_TRACK_HIGH_THRESHOLD = 0.25
    _DEFAULT_TRACK_LOW_THRESHOLD = 0.10
    _DEFAULT_NEW_TRACK_THRESHOLD = 0.25
    _DEFAULT_MATCH_THRESHOLD = 0.80
    _DEFAULT_FUSE_SCORE = True

    def __init__(
        self,
        trajectory_manager: Optional[TrajectoryManager] = None,
        track_high_threshold: float = _DEFAULT_TRACK_HIGH_THRESHOLD,
        track_low_threshold: float = _DEFAULT_TRACK_LOW_THRESHOLD,
        new_track_threshold: float = _DEFAULT_NEW_TRACK_THRESHOLD,
        track_buffer: int = config.TRACK_BUFFER,
        match_threshold: float = _DEFAULT_MATCH_THRESHOLD,
        fuse_score: bool = _DEFAULT_FUSE_SCORE,
    ) -> None:
        """Initialize ByteTrack and trajectory memory.

        Args:
            trajectory_manager: Optional external trajectory manager.
            track_high_threshold: First-stage association score threshold.
            track_low_threshold: Second-stage association score threshold.
            new_track_threshold: Score threshold for starting new tracks.
            track_buffer: Frames retained after a track becomes lost.
            match_threshold: ByteTrack matching threshold.
            fuse_score: Whether to fuse detection scores during matching.
        """
        self.track_buffer = track_buffer
        self._tracker_args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=track_high_threshold,
            track_low_thresh=track_low_threshold,
            new_track_thresh=new_track_threshold,
            track_buffer=track_buffer,
            match_thresh=match_threshold,
            fuse_score=fuse_score,
        )
        self.trajectory_manager = trajectory_manager or TrajectoryManager(
            max_lost_frames=track_buffer
        )
        self._tracker = self._load_bytetrack()
        self._frames_processed = 0
        self._detections_processed = 0
        self._tracks_emitted = 0
        self._last_update_latency_ms = 0.0
        self._last_bytetrack_latency_ms = 0.0

    def _load_bytetrack(self) -> Any:
        """Load and configure Ultralytics ByteTrack.

        Returns:
            Configured ByteTrack instance.
        """
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics ByteTrack is required for tracking."
            ) from exc

        return BYTETracker(self._tracker_args)

    def update(self, frame_data: FrameData) -> FrameData:
        """Update ByteTrack and store active tracks in ``FrameData``.

        Args:
            frame_data: Frame container with detector outputs.

        Returns:
            The same frame container with refreshed tracked objects.
        """
        update_start_time = perf_counter()
        self.trajectory_manager.frame_number = frame_data.frame_number
        previous_active_ids = set(self.trajectory_manager.active_tracks)
        previous_lost_ids = set(self.trajectory_manager.lost_tracks)
        detections = tuple(frame_data.detections)
        raw_tracks = self._update_bytetrack(detections, frame_data.image)
        matched_tracks = self._parse_tracks(raw_tracks, detections)
        active_ids = self._update_trajectory_manager(matched_tracks)

        for track_id in previous_active_ids - active_ids:
            self.trajectory_manager.mark_track_lost(track_id)
        for track_id in previous_lost_ids - active_ids:
            self.trajectory_manager.mark_track_lost(track_id)
        self.trajectory_manager.cleanup_tracks()

        frame_data.clear_tracks()
        for track in self._ordered_active_tracks(active_ids):
            frame_data.add_track(track)
        self._frames_processed += 1
        self._detections_processed += len(detections)
        self._tracks_emitted += frame_data.track_count
        self._last_update_latency_ms = self._elapsed_ms(update_start_time)
        return frame_data

    def reset(self) -> None:
        """Clear tracker and trajectory state between independent sequences."""
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()
        else:
            self._tracker = self._load_bytetrack()

        self.trajectory_manager.clear()
        self._frames_processed = 0
        self._detections_processed = 0
        self._tracks_emitted = 0
        self._last_update_latency_ms = 0.0
        self._last_bytetrack_latency_ms = 0.0

    def statistics(self) -> Dict[str, float]:
        """Return tracker, trajectory, and timing statistics."""
        stats = {
            key: float(value)
            for key, value in self.trajectory_manager.statistics().items()
        }
        stats.update(
            {
                "frames_processed": float(self._frames_processed),
                "detections_processed": float(self._detections_processed),
                "tracks_emitted": float(self._tracks_emitted),
                "last_update_latency_ms": self._last_update_latency_ms,
                "last_bytetrack_latency_ms": self._last_bytetrack_latency_ms,
            }
        )
        return stats

    def _update_bytetrack(
        self,
        detections: Sequence[Detection],
        image: np.ndarray,
    ) -> np.ndarray:
        """Convert detections and execute one ByteTrack update."""
        tracker_input = self._detections_to_bytetrack(detections)
        start_time = perf_counter()
        raw_tracks = self._tracker.update(tracker_input, img=image)
        self._last_bytetrack_latency_ms = self._elapsed_ms(start_time)
        return np.asarray(raw_tracks, dtype=np.float32)

    def _detections_to_bytetrack(
        self,
        detections: Sequence[Detection],
    ) -> _ByteTrackDetections:
        """Convert project detections into ByteTrack's input format.

        Args:
            detections: Detection objects produced by the detector.

        Returns:
            Results-like container exposing ``xywh``, ``conf``, and ``cls``.
        """
        if not detections:
            return _ByteTrackDetections(
                xywh=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.float32),
            )

        xywh = np.asarray(
            [self._bbox_to_xywh(detection.bbox) for detection in detections],
            dtype=np.float32,
        )
        confidence = np.asarray(
            [detection.confidence for detection in detections],
            dtype=np.float32,
        )
        class_ids = np.asarray(
            [detection.class_id for detection in detections],
            dtype=np.float32,
        )
        return _ByteTrackDetections(xywh, confidence, class_ids)

    def _parse_tracks(
        self,
        raw_tracks: Iterable[Any],
        detections: Sequence[Detection],
    ) -> List[_MatchedTrack]:
        """Decode ByteTrack output rows into typed matched tracks.

        Args:
            raw_tracks: ByteTrack output array.
            detections: Original detections from the current frame.

        Returns:
            Decoded active tracks.
        """
        tracks_array = np.asarray(raw_tracks, dtype=np.float32)
        if tracks_array.size == 0:
            return []
        tracks_array = np.atleast_2d(tracks_array)

        matched_tracks: List[_MatchedTrack] = []
        for row in tracks_array:
            if row.shape[0] < self._OUTPUT_MIN_COLUMNS:
                continue
            detection = self._detection_from_row(row, detections)
            bbox = self._sanitize_bbox(row)
            class_id = int(row[self._OUTPUT_CLASS_ID])
            class_name = (
                detection.class_name
                if detection is not None
                else str(class_id)
            )
            confidence = (
                detection.confidence
                if detection is not None
                else float(row[self._OUTPUT_SCORE])
            )
            matched_tracks.append(
                _MatchedTrack(
                    track_id=int(row[self._OUTPUT_TRACK_ID]),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    center=self._compute_center(bbox),
                )
            )
        return matched_tracks

    def _update_trajectory_manager(
        self,
        matched_tracks: Sequence[_MatchedTrack],
    ) -> Set[int]:
        """Update trajectory memory with active ByteTrack results.

        Args:
            matched_tracks: Active tracks decoded from ByteTrack.

        Returns:
            Set of active track identifiers in the current frame.
        """
        active_ids: Set[int] = set()
        for matched_track in matched_tracks:
            managed_track = self._update_or_create_track(matched_track)
            active_ids.add(managed_track.track_id)
        return active_ids

    def _update_or_create_track(
        self,
        matched_track: _MatchedTrack,
    ) -> TrackedObject:
        """Update an existing object in place or create a new managed track."""
        track = self.trajectory_manager.active_tracks.get(
            matched_track.track_id
        )
        if (
            track is None
            and matched_track.track_id in self.trajectory_manager.lost_tracks
        ):
            track = self.trajectory_manager.restore_track(
                matched_track.track_id
            )
        if track is None:
            track = self._to_tracked_object(matched_track)
            return self.trajectory_manager.add_track(track)

        self._apply_track_update(track, matched_track)
        self._record_track_hit(track)
        self.trajectory_manager.update_history(track.track_id, track.center)
        return track

    def _ordered_active_tracks(
        self,
        active_ids: Set[int],
    ) -> List[TrackedObject]:
        """Return active tracks in deterministic identifier order.

        Args:
            active_ids: Active track identifiers to emit.

        Returns:
            Managed active tracks sorted by identifier.
        """
        ordered_tracks: List[TrackedObject] = []
        for track_id in sorted(active_ids):
            track = self.trajectory_manager.active_tracks.get(track_id)
            if track is not None:
                ordered_tracks.append(track)
        return ordered_tracks

    def _to_tracked_object(
        self,
        matched_track: _MatchedTrack,
    ) -> TrackedObject:
        """Create a project-native tracked object from a matched track."""
        tracked_object = TrackedObject(
            track_id=matched_track.track_id,
            class_name=matched_track.class_name,
            confidence=matched_track.confidence,
            bbox=matched_track.bbox,
            center=matched_track.center,
        )
        tracked_object.update_frame(self.trajectory_manager.frame_number)
        return tracked_object

    def _apply_track_update(
        self,
        track: TrackedObject,
        matched_track: _MatchedTrack,
    ) -> None:
        """Mutate an existing tracked object with current association data."""
        track.class_name = matched_track.class_name
        track.update_bbox(matched_track.bbox)
        track.update_center(matched_track.center)
        track.update_confidence(matched_track.confidence)
        track.update_frame(self.trajectory_manager.frame_number)
        track.activate()

    @staticmethod
    def _record_track_hit(track: TrackedObject) -> None:
        """Update lifecycle counters stored on an existing track object."""
        setattr(track, "age", int(getattr(track, "age", 0)) + 1)
        setattr(track, "hits", int(getattr(track, "hits", 0)) + 1)
        setattr(track, "missed_frames", 0)

    def _detection_from_row(
        self,
        row: np.ndarray,
        detections: Sequence[Detection],
    ) -> Optional[Detection]:
        """Return source detection referenced by a ByteTrack output row."""
        detection_index = int(row[self._OUTPUT_DETECTION_INDEX])
        if detection_index < 0 or detection_index >= len(detections):
            return None
        return detections[detection_index]

    def _bbox_to_xywh(
        self,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, float, float, float]:
        """Convert ``xyxy`` box to ByteTrack ``xywh`` format."""
        x_min, y_min, x_max, y_max = bbox
        width = float(x_max - x_min)
        height = float(y_max - y_min)
        center_x = float(x_min) + width / self._CENTER_DIVISOR
        center_y = float(y_min) + height / self._CENTER_DIVISOR
        return center_x, center_y, width, height

    def _compute_center(
        self,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, float]:
        """Compute center point from an ``xyxy`` bounding box."""
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / self._CENTER_DIVISOR
        center_y = (y_min + y_max) / self._CENTER_DIVISOR
        return center_x, center_y

    @classmethod
    def _sanitize_bbox(cls, row: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert ByteTrack row coordinates to an integer ``xyxy`` box."""
        return (
            int(round(float(row[cls._BBOX_X_MIN]))),
            int(round(float(row[cls._BBOX_Y_MIN]))),
            int(round(float(row[cls._BBOX_X_MAX]))),
            int(round(float(row[cls._BBOX_Y_MAX]))),
        )

    @staticmethod
    def _elapsed_ms(start_time: float) -> float:
        """Return elapsed wall-clock time in milliseconds."""
        return (perf_counter() - start_time) * 1000.0
