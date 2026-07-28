"""
Trajectory memory management for temporal risk prediction.

The manager owns only tracked-object lifecycle and trajectory history.
Detection, tracking association, prediction, risk scoring, and visualization
remain isolated in their dedicated modules.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import config
from tracked_object import TrackedObject


class TrackState(Enum):
    """Lifecycle state for a managed road-user track."""

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    LOST = "LOST"
    FINISHED = "FINISHED"


class TrajectoryManager:
    """Central memory manager for :class:`TrackedObject` instances."""

    _FRAME_START = 0
    _FRAME_STEP = 1
    _COUNT_START = 0
    _INITIAL_AGE = 1
    _INITIAL_HITS = 1
    _INITIAL_MISSES = 0
    _NO_ACTIVE_LIMIT = 0

    def __init__(
        self,
        max_lost_frames: int = config.TRACK_BUFFER,
        max_history_length: int = config.MAX_HISTORY,
        min_history_length: int = config.MIN_SEQUENCE_LENGTH,
        max_active_tracks: Optional[int] = None,
    ) -> None:
        """Initialize track memories and lifecycle counters.

        Args:
            max_lost_frames: Consecutive misses allowed before finalization.
            max_history_length: Maximum retained history points per track.
            min_history_length: Minimum history points required to predict.
            max_active_tracks: Optional capacity for active tracks.
        """
        self._validate_limits(
            max_lost_frames,
            max_history_length,
            min_history_length,
            max_active_tracks,
        )
        self.active_tracks: Dict[int, TrackedObject] = {}
        self.lost_tracks: Dict[int, TrackedObject] = {}
        self.finished_tracks: Dict[int, TrackedObject] = {}
        self.frame_number = self._FRAME_START
        self.total_tracks = self._COUNT_START
        self.deleted_tracks = self._COUNT_START
        self.recovered_tracks = self._COUNT_START
        self.max_active_tracks = (
            max_active_tracks
            if max_active_tracks is not None
            else self._NO_ACTIVE_LIMIT
        )
        self._max_lost_frames = max_lost_frames
        self._max_history_length = max_history_length
        self._min_history_length = min_history_length

    def next_frame(self) -> int:
        """Advance the manager frame counter and return the new frame index."""
        self.frame_number += self._FRAME_STEP
        return self.frame_number

    def add_track(self, tracked_object: TrackedObject) -> TrackedObject:
        """Register a new object and move it through NEW to ACTIVE state.

        Args:
            tracked_object: New tracked object to retain.

        Returns:
            Registered active object.
        """
        track_id = tracked_object.track_id
        if self.exists(track_id):
            raise ValueError(f"Track {track_id} already exists.")

        self._enforce_active_capacity()
        self._initialize_track(tracked_object)
        self._set_state(tracked_object, TrackState.NEW)
        self._activate(tracked_object)
        self._set_state(tracked_object, TrackState.ACTIVE)
        self.active_tracks[track_id] = tracked_object
        self.total_tracks += self._FRAME_STEP
        return tracked_object

    def update_track(self, tracked_object: TrackedObject) -> TrackedObject:
        """Update active memory from a latest tracker observation.

        Args:
            tracked_object: Latest observation for a track.

        Returns:
            Managed object after update or registration.
        """
        track_id = tracked_object.track_id
        if track_id in self.active_tracks:
            track = self.active_tracks[track_id]
        elif track_id in self.lost_tracks:
            track = self.restore_track(track_id)
        else:
            return self.add_track(tracked_object)

        self._copy_observation(track, tracked_object)
        self._record_hit(track)
        self.update_history(track_id, track.center)
        return track

    def mark_track_lost(self, track_id: int) -> Optional[TrackedObject]:
        """Move an active track to LOST or increment an existing lost streak."""
        track = self.active_tracks.pop(track_id, None)
        if track is None:
            track = self.lost_tracks.get(track_id)
        if track is None:
            return None

        self._increment_misses(track)
        track.deactivate()
        self._set_state(track, TrackState.LOST)
        self.lost_tracks[track_id] = track
        return track

    def restore_track(self, track_id: int) -> TrackedObject:
        """Restore a LOST track to ACTIVE state."""
        if track_id not in self.lost_tracks:
            raise KeyError(f"Track {track_id} is not lost.")

        self._enforce_active_capacity()
        track = self.lost_tracks.pop(track_id)
        setattr(track, "missed_frames", self._INITIAL_MISSES)
        self._activate(track)
        self._set_state(track, TrackState.ACTIVE)
        self.recovered_tracks += self._FRAME_STEP
        return track

    def finish_track(self, track_id: int) -> Optional[TrackedObject]:
        """Move an active or lost track to FINISHED memory."""
        track = self.active_tracks.pop(track_id, None)
        if track is None:
            track = self.lost_tracks.pop(track_id, None)
        if track is None:
            return self.finished_tracks.get(track_id)

        track.deactivate()
        self._set_state(track, TrackState.FINISHED)
        self.finished_tracks[track_id] = track
        self.deleted_tracks += self._FRAME_STEP
        return track

    def cleanup_tracks(self) -> List[TrackedObject]:
        """Finish lost tracks that exceed the configured missed-frame buffer."""
        expired_ids = [
            track_id
            for track_id, track in self.lost_tracks.items()
            if self._misses(track) > self._max_lost_frames
        ]
        finished_tracks: List[TrackedObject] = []
        for track_id in expired_ids:
            track = self.finish_track(track_id)
            if track is not None:
                finished_tracks.append(track)
        return finished_tracks

    def update_history(
        self,
        track_id: int,
        point: Tuple[float, float],
    ) -> TrackedObject:
        """Append one trajectory point and refresh motion state."""
        track = self.get_track(track_id)
        if track is None:
            raise KeyError(f"Track {track_id} does not exist.")

        track.add_history(point)
        overflow = track.get_history_length() - self._max_history_length
        if overflow > self._COUNT_START:
            del track.history[:overflow]
        track.calculate_velocity()
        track.update_motion_state()
        return track

    def exists(self, track_id: int) -> bool:
        """Return whether a track id exists in any lifecycle memory."""
        return (
            track_id in self.active_tracks
            or track_id in self.lost_tracks
            or track_id in self.finished_tracks
        )

    def get_track(self, track_id: int) -> Optional[TrackedObject]:
        """Return a managed track by id, or ``None`` when absent."""
        return (
            self.active_tracks.get(track_id)
            or self.lost_tracks.get(track_id)
            or self.finished_tracks.get(track_id)
        )

    def get_all_tracks(self) -> List[TrackedObject]:
        """Return active, lost, and finished tracks."""
        return [
            *self.active_tracks.values(),
            *self.lost_tracks.values(),
            *self.finished_tracks.values(),
        ]

    def get_active_tracks(self) -> List[TrackedObject]:
        """Return active tracks."""
        return list(self.active_tracks.values())

    def get_lost_tracks(self) -> List[TrackedObject]:
        """Return lost tracks."""
        return list(self.lost_tracks.values())

    def get_finished_tracks(self) -> List[TrackedObject]:
        """Return finished tracks."""
        return list(self.finished_tracks.values())

    def get_tracks_by_class(self, class_name: str) -> List[TrackedObject]:
        """Return retained tracks matching a semantic class label."""
        return [
            track
            for track in self.get_all_tracks()
            if track.class_name == class_name
        ]

    def get_predictable_tracks(self) -> List[TrackedObject]:
        """Return active tracks with enough history for temporal prediction."""
        return [
            track
            for track in self.active_tracks.values()
            if track.has_enough_history(self._min_history_length)
        ]

    def statistics(self) -> Dict[str, int]:
        """Return deterministic lifecycle statistics."""
        return {
            "frame_number": self.frame_number,
            "active_tracks": len(self.active_tracks),
            "lost_tracks": len(self.lost_tracks),
            "finished_tracks": len(self.finished_tracks),
            "managed_tracks": len(self),
            "total_tracks": self.total_tracks,
            "deleted_tracks": self.deleted_tracks,
            "recovered_tracks": self.recovered_tracks,
            "predictable_tracks": len(self.get_predictable_tracks()),
            "max_active_tracks": self.max_active_tracks,
        }

    def clear(self) -> None:
        """Clear all track memories and reset manager counters."""
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.finished_tracks.clear()
        self.frame_number = self._FRAME_START
        self.total_tracks = self._COUNT_START
        self.deleted_tracks = self._COUNT_START
        self.recovered_tracks = self._COUNT_START

    def __len__(self) -> int:
        """Return the number of retained active, lost, and finished tracks."""
        return (
            len(self.active_tracks)
            + len(self.lost_tracks)
            + len(self.finished_tracks)
        )

    def __str__(self) -> str:
        """Return a concise manager summary string."""
        return (
            "TrajectoryManager("
            f"frame={self.frame_number}, "
            f"active={len(self.active_tracks)}, "
            f"lost={len(self.lost_tracks)}, "
            f"finished={len(self.finished_tracks)}, "
            f"total={self.total_tracks})"
        )

    @classmethod
    def _validate_limits(
        cls,
        max_lost_frames: int,
        max_history_length: int,
        min_history_length: int,
        max_active_tracks: Optional[int],
    ) -> None:
        """Validate lifecycle and history limits."""
        if max_lost_frames < cls._COUNT_START:
            raise ValueError("max_lost_frames must be non-negative.")
        if max_history_length < cls._INITIAL_AGE:
            raise ValueError("max_history_length must be positive.")
        if min_history_length < cls._INITIAL_AGE:
            raise ValueError("min_history_length must be positive.")
        if max_active_tracks is not None and max_active_tracks < cls._INITIAL_AGE:
            raise ValueError("max_active_tracks must be positive when provided.")

    def _initialize_track(self, track: TrackedObject) -> None:
        """Attach lifecycle metadata to a new track."""
        setattr(track, "age", self._INITIAL_AGE)
        setattr(track, "hits", self._INITIAL_HITS)
        setattr(track, "missed_frames", self._INITIAL_MISSES)
        track.update_frame(self.frame_number)
        if track.get_history_length() == self._COUNT_START:
            track.add_history(track.center)

    def _copy_observation(
        self,
        track: TrackedObject,
        observation: TrackedObject,
    ) -> None:
        """Copy observable fields from a latest observation."""
        track.update_bbox(observation.bbox)
        track.update_center(observation.center)
        track.update_confidence(observation.confidence)

        track.class_name = observation.class_name
        track.track_uuid = observation.track_uuid
        track.timestamp_ns = observation.timestamp_ns
        track.world_position = observation.world_position
        track.image_path = observation.image_path

        track.update_frame(self.frame_number)

    def _record_hit(self, track: TrackedObject) -> None:
        """Record a successful current-frame association."""
        setattr(track, "age", self._age(track) + self._FRAME_STEP)
        setattr(track, "hits", self._hits(track) + self._FRAME_STEP)
        setattr(track, "missed_frames", self._INITIAL_MISSES)
        self._activate(track)
        self._set_state(track, TrackState.ACTIVE)

    def _increment_misses(self, track: TrackedObject) -> None:
        """Increment a track's consecutive missed-frame counter."""
        setattr(track, "missed_frames", self._misses(track) + self._FRAME_STEP)

    def _activate(self, track: TrackedObject) -> None:
        """Activate a track and align its frame timestamp."""
        track.activate()
        track.update_frame(self.frame_number)

    @staticmethod
    def _set_state(track: TrackedObject, state: TrackState) -> None:
        """Attach lifecycle state metadata to a track."""
        setattr(track, "track_state", state)

    def _enforce_active_capacity(self) -> None:
        """Raise if adding an active track would exceed capacity."""
        if (
            self.max_active_tracks != self._NO_ACTIVE_LIMIT
            and len(self.active_tracks) >= self.max_active_tracks
        ):
            raise ValueError("Maximum number of active tracks reached.")

    @staticmethod
    def _age(track: TrackedObject) -> int:
        """Return associated-frame age stored on a track."""
        return int(getattr(track, "age", TrajectoryManager._INITIAL_MISSES))

    @staticmethod
    def _hits(track: TrackedObject) -> int:
        """Return successful association count stored on a track."""
        return int(getattr(track, "hits", TrajectoryManager._INITIAL_MISSES))

    @staticmethod
    def _misses(track: TrackedObject) -> int:
        """Return consecutive missed-frame count stored on a track."""
        return int(
            getattr(track, "missed_frames", TrajectoryManager._INITIAL_MISSES)
        )
