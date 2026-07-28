"""
YOLO-based object detection for temporal risk prediction.

The detector consumes :class:`FrameData`, appends :class:`Detection` objects,
and returns the same frame container for downstream ByteTrack processing.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import config

try:
    from frame_data import Detection, FrameData
except ModuleNotFoundError:
    from src.frame_data import Detection, FrameData


class Detector:
    """YOLO detector wrapper for autonomous-driving perception frames."""

    _BBOX_X_MIN = 0
    _BBOX_Y_MIN = 1
    _BBOX_X_MAX = 2
    _BBOX_Y_MAX = 3
    _CENTER_DIVISOR = 2.0
    _SECONDS_TO_MILLISECONDS = 1000.0
    _DEFAULT_LINE_THICKNESS = 2
    _DEFAULT_FONT_SCALE = 0.5
    _DEFAULT_TEXT_OFFSET = 6
    _DEFAULT_BOX_COLOR = (0, 255, 0)
    _DEFAULT_TEXT_COLOR = (255, 255, 255)

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        iou_threshold: float = config.IOU_THRESHOLD,
        target_classes: Optional[Sequence[str]] = None,
        class_ids: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize detector configuration and load the YOLO model.

        Args:
            model_path: Path or model name understood by Ultralytics YOLO.
            confidence_threshold: Minimum detection confidence.
            iou_threshold: IoU threshold used by YOLO non-maximum suppression.
            target_classes: Class names retained after inference.
            class_ids: Class identifiers retained after inference.
            device: Optional explicit inference device.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = tuple(target_classes or config.TARGET_CLASSES)
        self.device = device or self._select_device()
        self.model = self._load_model()
        configured_ids = class_ids
        if configured_ids is None:
            configured_ids = getattr(config, "CLASS_IDS", None)
        self.class_ids = self._resolve_class_ids(configured_ids)

    def _load_model(self) -> Any:
        """Load the configured YOLO model and bind it to the selected device.

        Returns:
            Loaded Ultralytics YOLO model.

        Raises:
            RuntimeError: If Ultralytics cannot be imported or the model fails
                to load.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics YOLO is required for detection."
            ) from exc

        try:
            model = YOLO(self.model_path)
            model.to(self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model '{self.model_path}'."
            ) from exc

        return model

    def detect(self, frame_data: FrameData) -> FrameData:
        """Run object detection and store results inside ``FrameData``.

        Args:
            frame_data: Frame container carrying the image to process.

        Returns:
            The same frame container with refreshed detections.
        """
        if frame_data.image is None:
            raise ValueError("FrameData.image must not be None.")

        frame_data.clear_detections()
        results = self.model.predict(
            source=frame_data.image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=sorted(self.class_ids) if self.class_ids else None,
            device=self.device,
            verbose=False,
        )

        for detection in self._parse_results(results):
            frame_data.add_detection(detection)

        return frame_data

    def _parse_results(self, results: Iterable[Any]) -> List[Detection]:
        """Convert YOLO output into project-native detection objects.

        Args:
            results: Iterable of Ultralytics result objects.

        Returns:
            Filtered detections in image coordinates.
        """
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = self._to_numpy(boxes.xyxy)
            confidences = self._to_numpy(boxes.conf)
            class_ids = self._to_numpy(boxes.cls).astype(int)

            for bbox_array, confidence, class_id in zip(
                xyxy,
                confidences,
                class_ids,
            ):
                class_name = self._class_name(class_id)
                if not self._filter_classes(class_id, class_name):
                    continue

                bbox = self._sanitize_bbox(bbox_array)
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=float(confidence),
                        bbox=bbox,
                        center=self._compute_center(bbox),
                    )
                )

        return detections

    def _compute_center(
        self,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, float]:
        """Compute center point from an ``(x1, y1, x2, y2)`` box."""
        x_min, y_min, x_max, y_max = bbox
        center_x = (x_min + x_max) / self._CENTER_DIVISOR
        center_y = (y_min + y_max) / self._CENTER_DIVISOR
        return center_x, center_y

    def _filter_classes(self, class_id: int, class_name: str) -> bool:
        """Return whether a YOLO class should be retained."""
        if self.class_ids and class_id not in self.class_ids:
            return False
        if self.target_classes and class_name not in self.target_classes:
            return False
        return True

    def draw(
        self,
        frame_data: FrameData,
        image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw detections on an image for debugging.

        Args:
            frame_data: Frame container with detections.
            image: Optional image to annotate instead of ``frame_data.image``.

        Returns:
            Annotated image copy.
        """
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for debug drawing.") from exc

        annotated = (image if image is not None else frame_data.image).copy()
        for detection in frame_data.detections:
            x_min, y_min, x_max, y_max = detection.bbox
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.rectangle(
                annotated,
                (x_min, y_min),
                (x_max, y_max),
                self._DEFAULT_BOX_COLOR,
                self._DEFAULT_LINE_THICKNESS,
            )
            cv2.putText(
                annotated,
                label,
                (x_min, max(y_min - self._DEFAULT_TEXT_OFFSET, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._DEFAULT_FONT_SCALE,
                self._DEFAULT_TEXT_COLOR,
                self._DEFAULT_LINE_THICKNESS,
                cv2.LINE_AA,
            )
        return annotated

    def benchmark(
        self,
        frame_data: FrameData,
        iterations: int = 20,
        warmup: int = 3,
    ) -> Dict[str, float]:
        """Benchmark detector latency on a representative frame.

        Args:
            frame_data: Frame container used for repeated inference.
            iterations: Timed inference iterations.
            warmup: Untimed warmup iterations.

        Returns:
            Latency and throughput statistics.
        """
        if iterations <= 0:
            raise ValueError("iterations must be positive.")
        if warmup < 0:
            raise ValueError("warmup must be non-negative.")

        for _ in range(warmup):
            self.detect(frame_data.copy_metadata())

        durations: List[float] = []
        for _ in range(iterations):
            benchmark_frame = frame_data.copy_metadata()
            start_time = perf_counter()
            self.detect(benchmark_frame)
            durations.append(perf_counter() - start_time)

        latencies_ms = np.asarray(durations, dtype=np.float64)
        latencies_ms *= self._SECONDS_TO_MILLISECONDS
        mean_latency = float(np.mean(latencies_ms))
        return {
            "iterations": float(iterations),
            "warmup": float(warmup),
            "mean_latency_ms": mean_latency,
            "min_latency_ms": float(np.min(latencies_ms)),
            "max_latency_ms": float(np.max(latencies_ms)),
            "std_latency_ms": float(np.std(latencies_ms)),
            "fps": self._SECONDS_TO_MILLISECONDS / mean_latency,
        }

    @staticmethod
    def _select_device() -> str:
        """Select CUDA when available and fall back to CPU otherwise."""
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_class_ids(
        self,
        configured_ids: Optional[Sequence[int]],
    ) -> Tuple[int, ...]:
        """Resolve configured or name-derived class ids."""
        if configured_ids:
            return tuple(int(class_id) for class_id in configured_ids)

        names = self._model_names()
        if not names:
            return tuple()

        return tuple(
            class_id
            for class_id, class_name in names.items()
            if class_name in self.target_classes
        )

    def _model_names(self) -> Dict[int, str]:
        """Return YOLO class-name mapping as ``{class_id: class_name}``."""
        names = getattr(self.model, "names", {})
        if isinstance(names, Mapping):
            return {int(class_id): str(name) for class_id, name in names.items()}
        if isinstance(names, Sequence):
            return {
                class_id: str(class_name)
                for class_id, class_name in enumerate(names)
            }
        return {}

    def _class_name(self, class_id: int) -> str:
        """Return class name for a YOLO class id."""
        return self._model_names().get(class_id, str(class_id))

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert tensors or arrays to detached CPU NumPy arrays."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            return value.numpy()
        return np.asarray(value)

    @classmethod
    def _sanitize_bbox(cls, bbox: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert a YOLO box into an integer ``(x1, y1, x2, y2)`` tuple."""
        return (
            int(round(float(bbox[cls._BBOX_X_MIN]))),
            int(round(float(bbox[cls._BBOX_Y_MIN]))),
            int(round(float(bbox[cls._BBOX_X_MAX]))),
            int(round(float(bbox[cls._BBOX_Y_MAX]))),
        )
