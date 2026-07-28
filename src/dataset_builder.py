"""
=========================================================
Project : Temporal Risk Prediction for Autonomous Driving
Author  : Jayanth K
Version : 1.0.0
=========================================================

Dataset Builder

This module reads the Argoverse 2 Sensor Dataset and
builds a processed dataset for temporal motion prediction.

Pipeline:

sensor/{train,val,test}/<log_id>/

annotations.feather + camera images

timestamp synchronization

dynamic object filtering

trajectory grouping (per track_uuid)

sliding window sequences (history + future)

normalized coordinates (relative to anchor)

processed_train.json
processed_val.json
processed_test.json
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config


# =========================================================
# Preprocessing Data Structures
# =========================================================

@dataclass
class CameraFrame:
    """
    Lightweight camera metadata for a single image frame.
    """

    timestamp_ns: int
    image_path: str


@dataclass
class TrajectoryPoint:
    """
    Lightweight representation of one ground-truth trajectory point
    from Argoverse 2 annotations.
    """

    timestamp_ns: int
    x: float
    y: float
    image_path: str
    track_uuid: str
    category: str
    frame_index: int
    timestamp_diff_ns: int


@dataclass
class TrainingSample:
    """
    One training sample for temporal motion prediction.

    history and future are stored as relative coordinates
    w.r.t. the last history point (anchor).
    """

    log_id: str
    image_path: str
    timestamp_ns: int
    track_uuid: str
    category: str

    # Absolute world position of the anchor used for normalization
    anchor_world_position: np.ndarray

    history: np.ndarray  # shape (H, 2), relative coords
    future: np.ndarray   # shape (F, 2), relative coords


# =========================================================
# Dataset Builder
# =========================================================

@dataclass
class ArgoverseDatasetBuilder:
    """
    Build a research-quality temporal motion prediction dataset
    from the Argoverse 2 Sensor logs.

    This class is responsible ONLY for converting raw AV2 data
    into compact training samples. It does not perform detection,
    tracking (ByteTrack), model training, or inference.
    """

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    # Root directory for AV2 sensor logs (e.g., datasets/av2/sensor)
    root: Path = field(default_factory=lambda: Path(config.DATASET_DIR))

    camera_name: str = field(default_factory=lambda: config.CAMERA_NAME)
    dynamic_categories: List[str] = field(
        default_factory=lambda: config.DATASET_TARGET_CLASSES
    )

    # Sequence lengths from config (history + future)
    history_length: int = field(default_factory=lambda: config.SEQUENCE_LENGTH)
    future_length: int = field(default_factory=lambda: config.PREDICTION_HORIZON)

    train_split: float = field(default_factory=lambda: config.TRAIN_SPLIT)
    val_split: float = field(default_factory=lambda: config.VAL_SPLIT)
    test_split: float = field(default_factory=lambda: config.TEST_SPLIT)

    # Directory for processed JSON dataset
    output_dir: Path = field(
        default_factory=lambda: Path(config.PROCESSED_DATASET_DIR)
    )

    random_seed: int = field(default_factory=lambda: config.RANDOM_SEED)

    # ---------------------------------------------------------
    # Internal State
    # ---------------------------------------------------------

    # Each entry is (split_name, log_path)
    logs: List[Tuple[str, Path]] = field(default_factory=list)
    logger: logging.Logger = field(init=False)

    annotations_by_log: Dict[str, pd.DataFrame] = field(default_factory=dict)
    camera_frames_by_log: Dict[str, List[CameraFrame]] = field(
        default_factory=dict
    )

    tracks_by_log: Dict[str, Dict[str, List[TrajectoryPoint]]] = field(
        default_factory=dict
    )

    samples: Dict[str, List[TrainingSample]] = field(default_factory=lambda: {
        "train": [],
        "val": [],
        "test": [],
    })

    def __post_init__(self) -> None:
        """Initialize logger, RNG, and normalize configuration paths."""
        self.root = self.root.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.logger.info("Initialized ArgoverseDatasetBuilder")
        self.logger.info(f"Root directory      : {self.root}")
        self.logger.info(f"Output directory    : {self.output_dir}")
        self.logger.info(f"Camera              : {self.camera_name}")
        self.logger.info(f"Dynamic categories  : {self.dynamic_categories}")
        self.logger.info(
            "Sequence lengths     : "
            f"history={self.history_length}, future={self.future_length}"
        )
        self.logger.info(
            "Dataset splits       : "
            f"train={self.train_split}, "
            f"val={self.val_split}, "
            f"test={self.test_split}"
        )

        self._validate_splits()
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def build(self) -> None:
        """
        Run the complete dataset construction pipeline:

            scan_logs()
            load_annotations()
            load_camera_images()
            synchronize_timestamps()
            filter_dynamic_objects()
            group_tracks()
            split_logs()
            create_sequences_per_split()
            save_dataset()
        """
        self.logger.info("Starting dataset build")

        self.scan_logs()
        self.logger.info("Stage 1: discovered %d logs", len(self.logs))

        self.load_annotations()
        self.logger.info(
            "Stage 2a: loaded annotations for %d logs",
            len(self.annotations_by_log),
        )

        self.load_camera_images()
        self.logger.info(
            "Stage 2b: prepared camera metadata for %d logs",
            len(self.camera_frames_by_log),
        )

        self.synchronize_timestamps()
        self.logger.info("Stage 2c: synchronized timestamps")

        self.filter_dynamic_objects()
        self.logger.info("Stage 3a: filtered dynamic objects")

        self.group_tracks()
        self.logger.info(
            "Stage 3b: grouped tracks for %d logs", len(self.tracks_by_log)
        )

        log_splits = self.split_logs()
        self.logger.info(
            "Stage 4: log splits "
            f"train={len(log_splits['train'])}, "
            f"val={len(log_splits['val'])}, "
            f"test={len(log_splits['test'])}"
        )

        self.create_sequences_per_split(log_splits)
        self.logger.info(
            "Stage 5: samples "
            f"train={len(self.samples['train'])}, "
            f"val={len(self.samples['val'])}, "
            f"test={len(self.samples['test'])}"
        )

        self.save_dataset()
        self.logger.info("Stage 6: saved processed dataset to %s", self.output_dir)

    # ---------------------------------------------------------
    # Stage 1: Directory Scanning
    # ---------------------------------------------------------

    def scan_logs(self) -> None:
        """
        Scan the Argoverse 2 sensor directory for logs, per split.

        Expected AV2 structure (per split):

            <root>/train/<log_id>/
                annotations.feather
                calibration/
                sensors/
                    cameras/
                        ring_front_center/
                            *.jpg

            <root>/val/<log_id>/
                ...

            <root>/test/<log_id>/
                ...

        Logs missing any critical component are skipped.
        """
        if not self.root.is_dir():
            raise FileNotFoundError(f"Argoverse root does not exist: {self.root}")

        self.logger.info("Scanning logs under %s", self.root)

        split_dirs = {
            "train": self.root / config.TRAIN_DIR,
            "val": self.root / config.VAL_DIR,
            "test": self.root / config.TEST_DIR,
        }

        self.logs.clear()
        split_counts = {"train": 0, "val": 0, "test": 0}

        for split_name, split_root in split_dirs.items():
            if not split_root.is_dir():
                self.logger.warning(
                    "Split directory missing for %s: %s", split_name, split_root
                )
                continue

            self.logger.info("Scanning %s logs in %s", split_name, split_root)

            log_dirs: List[Path] = [p for p in split_root.iterdir() if p.is_dir()]

            for log_dir in sorted(log_dirs):
                annotations_path = log_dir / "annotations.feather"
                calibration_dir = log_dir / "calibration"
                sensors_dir = log_dir / "sensors"
                camera_dir = sensors_dir / "cameras" / self.camera_name

                if not annotations_path.is_file():
                    self.logger.debug(
                        "[%s] Skipping %s (missing annotations.feather)",
                        split_name,
                        log_dir.name,
                    )
                    continue

                if not calibration_dir.is_dir():
                    self.logger.debug(
                        "[%s] Skipping %s (missing calibration/ directory)",
                        split_name,
                        log_dir.name,
                    )
                    continue

                if not sensors_dir.is_dir():
                    self.logger.debug(
                        "[%s] Skipping %s (missing sensors/ directory)",
                        split_name,
                        log_dir.name,
                    )
                    continue

                if not camera_dir.is_dir():
                    self.logger.debug(
                        "[%s] Skipping %s (missing camera directory: %s)",
                        split_name,
                        log_dir.name,
                        camera_dir,
                    )
                    continue

                self.logs.append((split_name, log_dir))
                split_counts[split_name] += 1

        total_logs = sum(split_counts.values())
        self.logger.info("Train logs        : %d", split_counts["train"])
        self.logger.info("Val logs          : %d", split_counts["val"])
        self.logger.info("Test logs         : %d", split_counts["test"])
        self.logger.info("Total logs        : %d", total_logs)

    # ---------------------------------------------------------
    # Stage 2a: Load Annotations
    # ---------------------------------------------------------

    def load_annotations(self) -> None:
        """
        Read `annotations.feather` for each valid log and store
        the resulting DataFrame in `annotations_by_log`.
        """
        self.annotations_by_log.clear()

        for split_name, log_dir in self.logs:
            log_id = f"{split_name}/{log_dir.name}"
            annotations_path = log_dir / "annotations.feather"

            self.logger.debug("Reading annotations for %s", log_id)
            df = pd.read_feather(annotations_path)

            required_cols = [
                "timestamp_ns",
                "track_uuid",
                "category",
                "tx_m",
                "ty_m",
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Annotations for {log_id} missing columns: {missing}"
                )

            self.annotations_by_log[log_id] = df

    # ---------------------------------------------------------
    # Stage 2b: Load Camera Image Metadata
    # ---------------------------------------------------------

    def load_camera_images(self) -> None:
        """
        Prepare per-log camera frame metadata.

        For each log, scan the configured camera directory and build a
        list of CameraFrame objects.
        """
        self.camera_frames_by_log.clear()

        for split_name, log_dir in self.logs:
            log_id = f"{split_name}/{log_dir.name}"
            camera_dir = (
                log_dir
                / "sensors"
                / "cameras"
                / self.camera_name
            )

            if not camera_dir.is_dir():
                self.logger.warning(
                    "Camera directory missing for %s: %s", log_id, camera_dir
                )
                continue

            frames: List[CameraFrame] = []

            for img_path in sorted(camera_dir.glob("*.jpg")):
                try:
                    timestamp_ns = int(img_path.stem)
                except ValueError:
                    timestamp_ns = 0

                frames.append(CameraFrame(
                    timestamp_ns=timestamp_ns,
                    image_path=str(img_path),
                ))

            frames.sort(key=lambda f: f.timestamp_ns)
            self.camera_frames_by_log[log_id] = frames

    # ---------------------------------------------------------
    # Stage 2c: Timestamp Synchronization
    # ---------------------------------------------------------

    def synchronize_timestamps(self) -> None:
        """
        Align annotation timestamps with camera frame timestamps.

        Adds `image_timestamp_ns` and `image_path` columns and
        computes timestamp differences for quality checks.
        """
        for log_id, df in self.annotations_by_log.items():
            frames = self.camera_frames_by_log.get(log_id, [])
            if not frames:
                self.logger.warning(
                    "No camera frames for %s; annotations remain unsynchronized",
                    log_id,
                )
                df = df.copy()
                df["image_timestamp_ns"] = df["timestamp_ns"]
                df["image_path"] = ""
                df["timestamp_diff_ns"] = 0
                self.annotations_by_log[log_id] = df
                continue

            frame_ts = [f.timestamp_ns for f in frames]

            image_timestamp_ns: List[int] = []
            image_paths: List[str] = []
            timestamp_diff_ns: List[int] = []

            for ts in df["timestamp_ns"].values:
                idx = self._nearest_index(frame_ts, ts)
                nearest_ts = frames[idx].timestamp_ns
                nearest_path = frames[idx].image_path
                image_timestamp_ns.append(nearest_ts)
                image_paths.append(nearest_path)
                timestamp_diff_ns.append(int(nearest_ts) - int(ts))

            df = df.copy()
            df["image_timestamp_ns"] = image_timestamp_ns
            df["image_path"] = image_paths
            df["timestamp_diff_ns"] = timestamp_diff_ns
            self.annotations_by_log[log_id] = df

    @staticmethod
    def _nearest_index(sorted_list: List[int], value: int) -> int:
        import bisect

        if not sorted_list:
            return 0

        pos = bisect.bisect_left(sorted_list, value)
        if pos == 0:
            return 0
        if pos == len(sorted_list):
            return len(sorted_list) - 1
        before = sorted_list[pos - 1]
        after = sorted_list[pos]
        if value - before <= after - value:
            return pos - 1
        return pos

    # ---------------------------------------------------------
    # Stage 3a: Filter Dynamic Objects
    # ---------------------------------------------------------

    def filter_dynamic_objects(self) -> None:
        """
        Keep only dynamic object categories specified in configuration.
        """
        for log_id, df in self.annotations_by_log.items():
            mask = df["category"].isin(self.dynamic_categories)
            dyn_df = df.loc[mask].copy()
            self.annotations_by_log[log_id] = dyn_df

    # ---------------------------------------------------------
    # Stage 3b: Group Tracks
    # ---------------------------------------------------------

    def group_tracks(self) -> None:
        """
        Group dynamic annotations by track_uuid and build
        per-track lists of TrajectoryPoint instances.
        """
        self.tracks_by_log.clear()

        for log_id, df in self.annotations_by_log.items():
            tracks: Dict[str, List[TrajectoryPoint]] = {}

            df_sorted = df.sort_values("timestamp_ns").reset_index(drop=True)

            for frame_index, (_, row) in enumerate(df_sorted.iterrows()):
                track_uuid = str(row["track_uuid"])
                category = str(row["category"])
                timestamp_ns = int(row["timestamp_ns"])
                tx_m = float(row["tx_m"])
                ty_m = float(row["ty_m"])
                image_path = str(row.get("image_path", ""))
                ts_diff = int(row.get("timestamp_diff_ns", 0))

                point = TrajectoryPoint(
                    timestamp_ns=timestamp_ns,
                    x=tx_m,
                    y=ty_m,
                    image_path=image_path,
                    track_uuid=track_uuid,
                    category=category,
                    frame_index=frame_index,
                    timestamp_diff_ns=ts_diff,
                )

                tracks.setdefault(track_uuid, []).append(point)

            self.tracks_by_log[log_id] = tracks

    # ---------------------------------------------------------
    # Stage 4: Split Logs (no leakage)
    # ---------------------------------------------------------

    def split_logs(self) -> Dict[str, List[str]]:
        """
        Split logs into train / val / test.

        If official AV2 val/test folders exist, preserve them.

        Otherwise, automatically create a deterministic random split
        from the available training logs.
        """
        split_to_logs = {
            "train": [],
            "val": [],
            "test": [],
        }

        for split_name, log_dir in self.logs:
            log_id = f"{split_name}/{log_dir.name}"
            split_to_logs[split_name].append(log_id)

        # ------------------------------------------------------
        # Fallback:
        # only train folder exists
        # ------------------------------------------------------
        if (
            len(split_to_logs["val"]) == 0
            and len(split_to_logs["test"]) == 0
        ):
            self.logger.warning(
                "Official val/test splits not found. "
                "Creating deterministic train/val/test split."
            )

            train_logs = split_to_logs["train"].copy()

            random.Random(self.random_seed).shuffle(train_logs)

            n = len(train_logs)

            n_train = int(self.train_split * n)
            n_val = int(self.val_split * n)

            split_to_logs["train"] = train_logs[:n_train]
            split_to_logs["val"] = train_logs[n_train:n_train + n_val]
            split_to_logs["test"] = train_logs[n_train + n_val:]

        return split_to_logs

    # ---------------------------------------------------------
    # Stage 5: Create Sequences Per Split (normalized)
    # ---------------------------------------------------------

    def create_sequences_per_split(
        self,
        log_splits: Dict[str, List[str]],
    ) -> None:
        """
        For each split, generate TrainingSample objects using
        a sliding window over tracks in the corresponding logs.

        Trajectories are stored in coordinates relative to
        the last history point (anchor).
        """
        self.samples = {"train": [], "val": [], "test": []}

        total_len = self.history_length + self.future_length
        if total_len <= 0:
            raise ValueError("history_length + future_length must be positive.")

        for split_name, log_ids in log_splits.items():
            split_samples: List[TrainingSample] = []

            for log_id in log_ids:
                tracks = self.tracks_by_log.get(log_id, {})
                for track_uuid, obs_list in tracks.items():
                    if len(obs_list) < total_len:
                        continue

                    for start in range(
                        0,
                        len(obs_list) - total_len + 1,
                        config.SEQUENCE_STRIDE,
                    ):
                        window = obs_list[start: start + total_len]
                        history_points = window[: self.history_length]
                        future_points = window[self.history_length:]

                        anchor = history_points[-1]
                        ax, ay = anchor.x, anchor.y

                        anchor_world_position = np.array(
                            [ax, ay],
                            dtype=np.float32,
                        )

                        history_xy = np.array(
                            [[p.x - ax, p.y - ay] for p in history_points],
                            dtype=np.float32,
                        )
                        future_xy = np.array(
                            [[p.x - ax, p.y - ay] for p in future_points],
                            dtype=np.float32,
                        )

                        sample = TrainingSample(
                            log_id=log_id,
                            image_path=anchor.image_path,
                            timestamp_ns=anchor.timestamp_ns,
                            track_uuid=track_uuid,
                            category=anchor.category,
                            anchor_world_position=anchor_world_position,
                            history=history_xy,
                            future=future_xy,
                        )
                        split_samples.append(sample)

            self.samples[split_name] = split_samples

    # ---------------------------------------------------------
    # Stage 6: Save Processed Dataset
    # ---------------------------------------------------------

    def save_dataset(self) -> None:
        """
        Save processed dataset to disk as JSON-like dicts.

        Each split is written to:

            processed_train.json
            processed_val.json
            processed_test.json

        Debug-friendly; production code can switch to .npz/.pt
        with minimal changes.
        """
        import json

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_samples in self.samples.items():
            out_path = self.output_dir / f"processed_{split_name}.json"
            as_dicts = [self._sample_to_serializable(s) for s in split_samples]
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(as_dicts, f, ensure_ascii=False, indent=2)

        meta = self.statistics()
        meta_path = self.output_dir / "dataset_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _sample_to_serializable(sample: TrainingSample) -> Dict:
        d = asdict(sample)
        d["anchor_world_position"] = sample.anchor_world_position.tolist()
        d["history"] = sample.history.tolist()
        d["future"] = sample.future.tolist()
        return d

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def statistics(self) -> Dict[str, int]:
        """
        Return basic statistics about the constructed dataset,
        including number of tracks and per-split sample counts.
        """
        num_tracks = sum(
            len(tracks) for tracks in self.tracks_by_log.values()
        )

        return {
            "num_logs": len(self.logs),
            "num_annotation_logs": len(self.annotations_by_log),
            "num_camera_logs": len(self.camera_frames_by_log),
            "num_tracks_logs": len(self.tracks_by_log),
            "num_tracks": num_tracks,
            "num_train_samples": len(self.samples.get("train", [])),
            "num_val_samples": len(self.samples.get("val", [])),
            "num_test_samples": len(self.samples.get("test", [])),
            "history_length": self.history_length,
            "future_length": self.future_length,
        }

    # ---------------------------------------------------------
    # Internal Helpers
    # ---------------------------------------------------------

    def _validate_splits(self) -> None:
        total = self.train_split + self.val_split + self.test_split
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Dataset splits must sum to 1.0, got {total:.4f} "
                f"(train={self.train_split}, "
                f"val={self.val_split}, "
                f"test={self.test_split})"
            )

        if not (0.0 < self.train_split < 1.0):
            raise ValueError("train_split must be in (0, 1).")
        if not (0.0 <= self.val_split < 1.0):
            raise ValueError("val_split must be in [0, 1).")
        if not (0.0 <= self.test_split < 1.0):
            raise ValueError("test_split must be in [0, 1).")


# =========================================================
# Entry Point
# ==========================================================

def main() -> None:
    builder = ArgoverseDatasetBuilder()
    builder.build()


if __name__ == "__main__":
    main()
