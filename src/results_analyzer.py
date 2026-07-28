"""
=========================================================
Project : Temporal Risk Prediction for Autonomous Driving
Author  : Jayanth K
Version : 2.1.0
=========================================================

Results Analyzer / Visualizer V2

This module loads trajectory prediction and risk results
and renders an MP4 dashboard visualization with:

    - camera context panel
    - bird's-eye-view (BEV) trajectory panel
    - statistics panel
    - per-track metrics panel
    - footer legend / credits

Pipeline

TrackedObject states + trajectories
        ↓
ResultsVisualizer
        ↓
Dashboard rendering
        ↓
results.mp4
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

import config
from tracked_object import TrackedObject


# =========================================================
# Utility Structures
# ==========================================================

@dataclass
class FrameResult:
    frame_index: int
    image_path: str
    tracks: List[TrackedObject]


@dataclass
class DashboardLayout:
    total_width: int
    total_height: int

    header_height: int
    footer_height: int

    camera_x: int
    camera_y: int
    camera_width: int
    camera_height: int

    lower_y: int
    lower_height: int

    bev_x: int
    bev_y: int
    bev_width: int
    bev_height: int

    right_x: int
    right_y: int
    right_width: int
    right_height: int

    stats_x: int
    stats_y: int
    stats_width: int
    stats_height: int

    metrics_x: int
    metrics_y: int
    metrics_width: int
    metrics_height: int

    margin: int
    padding: int
    panel_gap: int


@dataclass
class BEVTransform:
    scale: float
    origin_x: int
    origin_y: int
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    meters_per_division: float


# =========================================================
# Results Visualizer / Analyzer
# ==========================================================

class ResultsVisualizer:
    def __init__(
        self,
        results_path: str | Path,
        output_video_dir: str | Path | None = None,
    ) -> None:
        self.results_path = Path(results_path)

        if output_video_dir is None:
            output_video_dir = config.OUTPUT_VIDEOS_DIR
        self.output_video_dir = Path(output_video_dir)

        self.font_scale = config.VIS_FONT_SCALE
        self.font_thickness = config.VIS_FONT_THICKNESS
        self.line_thickness = config.VIS_LINE_THICKNESS
        self.circle_radius = config.VIS_CIRCLE_RADIUS
        self.video_fps = config.VIS_VIDEO_FPS

        self.color_history = config.VIS_COLOR_HISTORY
        self.color_gt = config.VIS_COLOR_GT
        self.color_pred = config.VIS_COLOR_PRED

        self.color_risk_low = config.VIS_COLOR_RISK_LOW
        self.color_risk_mid = config.VIS_COLOR_RISK_MID
        self.color_risk_high = config.VIS_COLOR_RISK_HIGH

        self.color_text_bg = config.VIS_COLOR_TEXT_BG
        self.color_text_fg = config.VIS_COLOR_TEXT_FG

        self.legend_margin = config.VIS_LEGEND_MARGIN
        self.legend_line_length = config.VIS_LEGEND_LINE_LENGTH

        self.risk_low_threshold = config.LOW_RISK_THRESHOLD
        self.risk_mid_threshold = config.MEDIUM_RISK_THRESHOLD

        self.bg_color = (15, 15, 18)
        self.header_color = (20, 22, 28)
        self.footer_color = (18, 20, 24)
        self.panel_color = (28, 30, 36)
        self.panel_color_alt = (34, 37, 44)
        self.panel_border_color = (78, 82, 94)

        self.grid_color = (64, 68, 78)
        self.axis_color = (190, 195, 205)
        self.muted_text_color = (180, 185, 195)
        self.accent_blue = (255, 120, 40)
        self.camera_overlay_color = (0, 0, 0)

        self.frames: List[FrameResult] = []
        self.layout: DashboardLayout | None = None
        self.bev_transform: BEVTransform | None = None

    # =====================================================
    # Public API
    # =====================================================

    def load_results(self) -> None:
        if not self.results_path.exists():
            raise FileNotFoundError(self.results_path)

        with self.results_path.open("r", encoding="utf-8") as f:
            raw_frames = json.load(f)

        frames: List[FrameResult] = []

        for raw in raw_frames:
            frame_index = int(raw["frame_index"])
            image_path = str(raw["image_path"])
            tracks: List[TrackedObject] = []

            for t in raw["tracks"]:
                obj = TrackedObject(
                    track_id=int(t["track_id"]),
                    track_uuid=str(t["track_uuid"]),
                    class_name=str(t["class_name"]),
                    confidence=float(t["confidence"]),
                    bbox=tuple(t["bbox"]),
                    center=tuple(t["center"]),
                    world_position=tuple(t.get("world_position", (0.0, 0.0))),
                    anchor_world_position=tuple(
                        t.get("anchor_world_position", (0.0, 0.0))
                    ),
                    velocity=tuple(t.get("velocity", (0.0, 0.0))),
                )

                obj.history = [tuple(p) for p in t.get("history", [])]
                obj.future_prediction = [tuple(p) for p in t.get("future_prediction", [])]
                obj.future_ground_truth = [tuple(p) for p in t.get("future_ground_truth", [])]

                obj.risk_score = float(t.get("risk_score", 0.0))
                obj.risk_level = str(t.get("risk_level", "LOW"))
                obj.ttc = float(t.get("ttc", float("inf")))

                obj.ade = float(t.get("ade", 0.0))
                obj.fde = float(t.get("fde", 0.0))

                tracks.append(obj)

            frames.append(
                FrameResult(
                    frame_index=frame_index,
                    image_path=image_path,
                    tracks=tracks,
                )
            )

        self.frames = frames

    def render(self, output_name: str = "result.mp4") -> Path:
        if not self.frames:
            raise RuntimeError("No frames loaded; call load_results() first.")

        self.output_video_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_video_dir / output_name

        first_frame_image = cv2.imread(self.frames[0].image_path)
        if first_frame_image is None:
            raise RuntimeError(
                f"Failed to read first frame image: {self.frames[0].image_path}"
            )

        image_height, image_width = first_frame_image.shape[:2]

        self.layout = self._compute_dashboard_layout(image_width, image_height)
        self.bev_transform = self._compute_global_bev_transform()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path.as_posix(),
            fourcc,
            self.video_fps,
            (self.layout.total_width, self.layout.total_height),
        )

        for frame in self.frames:
            dashboard = self.render_frame(frame)
            writer.write(dashboard)

        writer.release()
        return output_path

    # =====================================================
    # Layout / Transform
    # =====================================================

    def _compute_dashboard_layout(
        self,
        image_width: int,
        image_height: int,
    ) -> DashboardLayout:
        margin = 20
        padding = 16
        panel_gap = 20

        header_height = 86
        footer_height = 72

        camera_width = image_width
        camera_height = image_height

        lower_height = max(470, int(image_height * 0.47))
        bev_width = int(image_width * 0.58)
        bev_height = lower_height

        right_width = image_width - bev_width - panel_gap
        right_height = lower_height

        stats_height = 160
        metrics_height = right_height - stats_height - panel_gap

        total_width = margin * 2 + image_width
        total_height = (
            margin
            + header_height
            + panel_gap
            + camera_height
            + panel_gap
            + lower_height
            + panel_gap
            + footer_height
            + margin
        )

        camera_x = margin
        camera_y = margin + header_height + panel_gap

        lower_y = camera_y + camera_height + panel_gap

        bev_x = margin
        bev_y = lower_y

        right_x = bev_x + bev_width + panel_gap
        right_y = lower_y

        stats_x = right_x
        stats_y = right_y
        stats_width = right_width
        stats_height = stats_height

        metrics_x = right_x
        metrics_y = stats_y + stats_height + panel_gap
        metrics_width = right_width
        metrics_height = metrics_height

        return DashboardLayout(
            total_width=total_width,
            total_height=total_height,
            header_height=header_height,
            footer_height=footer_height,
            camera_x=camera_x,
            camera_y=camera_y,
            camera_width=camera_width,
            camera_height=camera_height,
            lower_y=lower_y,
            lower_height=lower_height,
            bev_x=bev_x,
            bev_y=bev_y,
            bev_width=bev_width,
            bev_height=bev_height,
            right_x=right_x,
            right_y=right_y,
            right_width=right_width,
            right_height=right_height,
            stats_x=stats_x,
            stats_y=stats_y,
            stats_width=stats_width,
            stats_height=stats_height,
            metrics_x=metrics_x,
            metrics_y=metrics_y,
            metrics_width=metrics_width,
            metrics_height=metrics_height,
            margin=margin,
            padding=padding,
            panel_gap=panel_gap,
        )

    def _compute_global_bev_transform(self) -> BEVTransform:
        if self.layout is None:
            raise RuntimeError("Dashboard layout must be computed first.")

        xs: List[float] = []
        ys: List[float] = []

        for frame in self.frames:
            for obj in frame.tracks:
                for x, y in self._get_absolute_history(obj):
                    xs.append(float(x))
                    ys.append(float(y))
                for x, y in self._get_absolute_prediction(obj):
                    xs.append(float(x))
                    ys.append(float(y))
                for x, y in self._get_absolute_ground_truth(obj):
                    xs.append(float(x))
                    ys.append(float(y))

                wx = (
                    obj.anchor_world_position[0]
                    + obj.world_position[0]
                )
                wy = (
                    obj.anchor_world_position[1]
                    + obj.world_position[1]
                )
                xs.append(float(wx))
                ys.append(float(wy))

        if not xs or not ys:
            min_x, max_x = -10.0, 10.0
            min_y, max_y = -10.0, 10.0
        else:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            if math.isclose(min_x, max_x):
                min_x -= 5.0
                max_x += 5.0
            if math.isclose(min_y, max_y):
                min_y -= 5.0
                max_y += 5.0

        pad_x = max(1.0, 0.1 * (max_x - min_x))
        pad_y = max(1.0, 0.1 * (max_y - min_y))

        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y

        usable_width = self.layout.bev_width - 2 * self.layout.padding
        usable_height = self.layout.bev_height - 70 - self.layout.padding

        span_x = max_x - min_x
        span_y = max_y - min_y

        scale_x = usable_width / max(span_x, 1e-6)
        scale_y = usable_height / max(span_y, 1e-6)
        scale = min(scale_x, scale_y)

        center_x_world = (min_x + max_x) / 2.0
        center_y_world = (min_y + max_y) / 2.0

        origin_x = int(self.layout.bev_width / 2 - center_x_world * scale)
        origin_y = int((70 + self.layout.bev_height) / 2 + center_y_world * scale / 2)

        approx_division = 5.0
        if scale > 0:
            for candidate in [1.0, 2.0, 5.0, 10.0, 20.0]:
                px = candidate * scale
                if 55 <= px <= 120:
                    approx_division = candidate
                    break

        return BEVTransform(
            scale=scale,
            origin_x=origin_x,
            origin_y=origin_y,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
            meters_per_division=approx_division,
        )

    # =====================================================
    # Frame Rendering
    # =====================================================

    def render_frame(self, frame: FrameResult) -> np.ndarray:
        if self.layout is None:
            raise RuntimeError("Layout not initialized.")
        if self.bev_transform is None:
            raise RuntimeError("BEV transform not initialized.")

        image = cv2.imread(frame.image_path)
        if image is None:
            image = np.zeros(
                (self.layout.camera_height, self.layout.camera_width, 3),
                dtype=np.uint8,
            )

        if image.shape[1] != self.layout.camera_width or image.shape[0] != self.layout.camera_height:
            image = cv2.resize(
                image,
                (self.layout.camera_width, self.layout.camera_height),
                interpolation=cv2.INTER_LINEAR,
            )

        header = self.draw_header(frame)
        camera_panel = self.draw_camera_panel(image, frame)
        bev_panel = self.draw_bev_panel(frame)
        stats_panel = self.draw_statistics_panel(frame)
        metrics_panel = self.draw_metrics_panel(frame)
        footer = self.draw_footer(frame)

        return self.compose_dashboard(
            header=header,
            camera_panel=camera_panel,
            bev_panel=bev_panel,
            stats_panel=stats_panel,
            metrics_panel=metrics_panel,
            footer=footer,
        )

    def draw_header(self, frame: FrameResult) -> np.ndarray:
        if self.layout is None:
            raise RuntimeError("Layout not initialized.")

        header = np.full(
            (self.layout.header_height, self.layout.total_width, 3),
            self.header_color,
            dtype=np.uint8,
        )

        title = "Learning-Based Temporal Motion Prediction for Risk-Aware Autonomous Driving"
        subtitle = (
            f"Author: Jayanth K  |  "
            f"Kyushu Institute of Technology  |  "
            f"Model: {config.MODEL_NAME}  |  "
            f"Dataset: {config.DATASET_NAME}  |  "
            f"Frame: {frame.frame_index:06d}"
        )

        cv2.putText(
            header,
            title,
            (self.layout.margin, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            self.color_text_fg,
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            header,
            subtitle,
            (self.layout.margin, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.muted_text_color,
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            header,
            (0, self.layout.header_height - 1),
            (self.layout.total_width, self.layout.header_height - 1),
            self.panel_border_color,
            1,
        )
        return header

    def draw_camera_panel(self, image: np.ndarray, frame: FrameResult) -> np.ndarray:
        if self.layout is None:
            raise RuntimeError("Layout not initialized.")

        panel = image.copy()

        avg_ade = self._safe_mean([obj.ade for obj in frame.tracks])
        avg_fde = self._safe_mean([obj.fde for obj in frame.tracks])

        low_risk = sum(1 for t in frame.tracks if t.risk_level.upper() == "LOW")
        med_risk = sum(1 for t in frame.tracks if t.risk_level.upper() == "MEDIUM")
        high_risk = sum(1 for t in frame.tracks if t.risk_level.upper() == "HIGH")

        overlay_h = 108
        overlay = panel.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (self.layout.camera_width, overlay_h),
            self.camera_overlay_color,
            thickness=-1,
        )
        panel = cv2.addWeighted(overlay, 0.55, panel, 0.45, 0)

        cv2.putText(
            panel,
            "CAMERA PANEL",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            self.color_text_fg,
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Model: {config.MODEL_NAME}   Dataset: {config.DATASET_NAME}",
            (18, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.color_text_fg,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Frame: {frame.frame_index:06d}   Tracks: {len(frame.tracks)}   Avg ADE: {avg_ade:.3f}   Avg FDE: {avg_fde:.3f}",
            (18, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.color_text_fg,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"Risk Summary  Low: {low_risk}   Medium: {med_risk}   High: {high_risk}",
            (18, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.muted_text_color,
            1,
            lineType=cv2.LINE_AA,
        )

        self._draw_panel_border(panel)
        return panel

    def draw_bev_panel(self, frame: FrameResult) -> np.ndarray:
        if self.layout is None or self.bev_transform is None:
            raise RuntimeError("Layout or transform not initialized.")

        bev = np.full(
            (self.layout.bev_height, self.layout.bev_width, 3),
            self.panel_color,
            dtype=np.uint8,
        )

        self._draw_panel_title(
            bev,
            "BEV PANEL",
            "History / Ground Truth / Prediction in model coordinate space"
        )

        plot_left = self.layout.padding
        plot_top = 56
        plot_right = self.layout.bev_width - self.layout.padding
        plot_bottom = self.layout.bev_height - self.layout.padding

        cv2.rectangle(
            bev,
            (plot_left, plot_top),
            (plot_right, plot_bottom),
            self.panel_color_alt,
            thickness=-1,
        )

        self._draw_bev_grid(bev, plot_left, plot_top, plot_right, plot_bottom)
        self._draw_bev_axes(bev, plot_left, plot_top, plot_right, plot_bottom)

        for obj in frame.tracks:
            hist_pts = self._trajectory_to_bev_pixels(
                self._get_absolute_history(obj)
            )
            gt_pts = self._trajectory_to_bev_pixels(
                self._get_absolute_ground_truth(obj)
            )
            pred_pts = self._trajectory_to_bev_pixels(
                self._get_absolute_prediction(obj)
            )

            self._draw_polyline_with_points(
                bev, hist_pts, self.color_history, max(2, self.circle_radius), self.line_thickness
            )
            self._draw_polyline_with_points(
                bev, gt_pts, self.color_gt, max(2, self.circle_radius), self.line_thickness
            )
            self._draw_polyline_with_points(
                bev, pred_pts, self.color_pred, max(2, self.circle_radius), self.line_thickness
            )

            anchor_pt = self._world_to_bev_pixel(
                (
                    obj.anchor_world_position[0]
                    + obj.world_position[0],
                    obj.anchor_world_position[1]
                    + obj.world_position[1],
                )
            )
            if anchor_pt is not None:
                self._draw_triangle_marker(bev, anchor_pt, self._risk_color(obj.risk_score))
                self._draw_velocity_arrow(bev, obj)

        self._draw_bev_scale(bev)
        self._draw_panel_border(bev)
        return bev

    def draw_statistics_panel(self, frame: FrameResult) -> np.ndarray:
        if self.layout is None:
            raise RuntimeError("Layout not initialized.")

        panel = np.full(
            (self.layout.stats_height, self.layout.stats_width, 3),
            self.panel_color,
            dtype=np.uint8,
        )

        self._draw_panel_title(panel, "STATISTICS", "Frame-level summary")

        vehicles = sum(1 for t in frame.tracks if "VEHICLE" in t.class_name.upper() or "TRUCK" in t.class_name.upper() or "BUS" in t.class_name.upper())
        pedestrians = sum(1 for t in frame.tracks if "PEDESTRIAN" in t.class_name.upper())
        cyclists = sum(1 for t in frame.tracks if "CYCLIST" in t.class_name.upper() or "BICYCLE" in t.class_name.upper())

        avg_ade = self._safe_mean([obj.ade for obj in frame.tracks])
        avg_fde = self._safe_mean([obj.fde for obj in frame.tracks])
        avg_risk = self._safe_mean([obj.risk_score for obj in frame.tracks])
        highest_risk = max((obj.risk_score for obj in frame.tracks), default=0.0)

        stats = [
            ("Tracks", str(len(frame.tracks))),
            ("Vehicles", str(vehicles)),
            ("Pedestrians", str(pedestrians)),
            ("Cyclists", str(cyclists)),
            ("Avg ADE", f"{avg_ade:.3f}"),
            ("Avg FDE", f"{avg_fde:.3f}"),
            ("Avg Risk", f"{avg_risk:.3f}"),
            ("Max Risk", f"{highest_risk:.3f}"),
        ]

        col1_x = 18
        col2_x = self.layout.stats_width // 2 + 10
        start_y = 68
        row_gap = 28

        for i, (label, value) in enumerate(stats[:4]):
            y = start_y + i * row_gap
            cv2.putText(panel, f"{label}:", (col1_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.muted_text_color, 1, lineType=cv2.LINE_AA)
            cv2.putText(panel, value, (col1_x + 110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, self.color_text_fg, 1, lineType=cv2.LINE_AA)

        for i, (label, value) in enumerate(stats[4:]):
            y = start_y + i * row_gap
            cv2.putText(panel, f"{label}:", (col2_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.muted_text_color, 1, lineType=cv2.LINE_AA)
            cv2.putText(panel, value, (col2_x + 110, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, self.color_text_fg, 1, lineType=cv2.LINE_AA)

        self._draw_panel_border(panel)
        return panel

    def draw_metrics_panel(self, frame: FrameResult) -> np.ndarray:
        if self.layout is None:
            raise RuntimeError("Layout not initialized.")

        panel = np.full(
            (self.layout.metrics_height, self.layout.metrics_width, 3),
            self.panel_color,
            dtype=np.uint8,
        )

        self._draw_panel_title(panel, "TRACK METRICS", "Per-object cards")

        if not frame.tracks:
            cv2.putText(
                panel,
                "No tracked objects available in this frame.",
                (self.layout.padding, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.color_text_fg,
                1,
                lineType=cv2.LINE_AA,
            )
            self._draw_panel_border(panel)
            return panel

        card_h = 96
        gap = 12
        start_y = 58
        max_cards = max(1, (self.layout.metrics_height - start_y - self.layout.padding) // (card_h + gap))

        for idx, obj in enumerate(frame.tracks[:max_cards]):
            y1 = start_y + idx * (card_h + gap)
            y2 = y1 + card_h
            x1 = self.layout.padding
            x2 = self.layout.metrics_width - self.layout.padding

            cv2.rectangle(panel, (x1, y1), (x2, y2), self.panel_color_alt, thickness=-1)
            cv2.rectangle(panel, (x1, y1), (x1 + 8, y2), self._risk_color(obj.risk_score), thickness=-1)

            speed = self._speed_magnitude(obj.velocity)
            ttc_str = "inf" if math.isinf(obj.ttc) else f"{obj.ttc:.2f}s"

            cv2.putText(
                panel,
                f"{obj.class_name}  |  ID {obj.track_id}",
                (x1 + 18, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                self.color_text_fg,
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"Risk: {obj.risk_level.upper()} ({obj.risk_score:.2f})",
                (x1 + 18, y1 + 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                self._risk_color(obj.risk_score),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"ADE: {obj.ade:.3f}   FDE: {obj.fde:.3f}   Speed: {speed:.2f}   TTC: {ttc_str}",
                (x1 + 18, y1 + 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                self.muted_text_color,
                1,
                lineType=cv2.LINE_AA,
            )

        if len(frame.tracks) > max_cards:
            extra = len(frame.tracks) - max_cards
            cv2.putText(
                panel,
                f"+ {extra} more tracks not shown",
                (self.layout.padding, self.layout.metrics_height - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.muted_text_color,
                1,
                lineType=cv2.LINE_AA,
            )

        self._draw_panel_border(panel)
        return panel

    def draw_footer(self, frame: FrameResult) -> np.ndarray:
        if self.layout is None or self.bev_transform is None:
            raise RuntimeError("Layout or transform not initialized.")

        footer = np.full(
            (self.layout.footer_height, self.layout.total_width, 3),
            self.footer_color,
            dtype=np.uint8,
        )

        self._draw_legend(footer)

        scale_text = f"Scale: 1 division = {self.bev_transform.meters_per_division:.0f} m"
        credit_text = "Developed by Jayanth K • Kyushu Institute of Technology"

        cv2.putText(
            footer,
            scale_text,
            (self.layout.total_width - 520, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.color_text_fg,
            1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            footer,
            credit_text,
            (self.layout.total_width - 520, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.muted_text_color,
            1,
            lineType=cv2.LINE_AA,
        )

        cv2.line(
            footer,
            (0, 0),
            (self.layout.total_width, 0),
            self.panel_border_color,
            1,
        )
        return footer

    def compose_dashboard(
        self,
        header: np.ndarray,
        camera_panel: np.ndarray,
        bev_panel: np.ndarray,
        stats_panel: np.ndarray,
        metrics_panel: np.ndarray,
        footer: np.ndarray,
    ) -> np.ndarray:
        if self.layout is None:
            raise RuntimeError("Layout not initialized.")

        dashboard = np.full(
            (self.layout.total_height, self.layout.total_width, 3),
            self.bg_color,
            dtype=np.uint8,
        )

        dashboard[0:self.layout.header_height, :] = header

        cy1 = self.layout.camera_y
        cy2 = cy1 + self.layout.camera_height
        cx1 = self.layout.camera_x
        cx2 = cx1 + self.layout.camera_width
        dashboard[cy1:cy2, cx1:cx2] = camera_panel

        by1 = self.layout.bev_y
        by2 = by1 + self.layout.bev_height
        bx1 = self.layout.bev_x
        bx2 = bx1 + self.layout.bev_width
        dashboard[by1:by2, bx1:bx2] = bev_panel

        sy1 = self.layout.stats_y
        sy2 = sy1 + self.layout.stats_height
        sx1 = self.layout.stats_x
        sx2 = sx1 + self.layout.stats_width
        dashboard[sy1:sy2, sx1:sx2] = stats_panel

        my1 = self.layout.metrics_y
        my2 = my1 + self.layout.metrics_height
        mx1 = self.layout.metrics_x
        mx2 = mx1 + self.layout.metrics_width
        dashboard[my1:my2, mx1:mx2] = metrics_panel

        fy1 = self.layout.total_height - self.layout.margin - self.layout.footer_height
        fy2 = fy1 + self.layout.footer_height
        dashboard[fy1:fy2, :] = footer

        return dashboard

    # =====================================================
    # Drawing Helpers
    # =====================================================

    def _draw_legend(self, image: np.ndarray) -> None:
        y = 28
        x = self.legend_margin + 12

        entries = [
            ("History", self.color_history),
            ("Ground Truth", self.color_gt),
            ("Prediction", self.color_pred),
            ("Low Risk", self.color_risk_low),
            ("Medium Risk", self.color_risk_mid),
            ("High Risk", self.color_risk_high),
        ]

        for label, color in entries:
            cv2.line(
                image,
                (x, y),
                (x + self.legend_line_length, y),
                color,
                thickness=self.line_thickness,
            )
            cv2.circle(
                image,
                (x + self.legend_line_length // 2, y),
                max(2, self.circle_radius),
                color,
                thickness=-1,
            )
            cv2.putText(
                image,
                label,
                (x + self.legend_line_length + 8, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                self.color_text_fg,
                1,
                lineType=cv2.LINE_AA,
            )
            x += self.legend_line_length + 8 + 118

    def _draw_panel_title(self, panel: np.ndarray, title: str, subtitle: str) -> None:
        cv2.putText(
            panel,
            title,
            (16, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            self.color_text_fg,
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            subtitle,
            (16, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            self.muted_text_color,
            1,
            lineType=cv2.LINE_AA,
        )

    def _draw_panel_border(self, panel: np.ndarray) -> None:
        h, w = panel.shape[:2]
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), self.panel_border_color, 1)

    def _draw_bev_grid(self, bev: np.ndarray, left: int, top: int, right: int, bottom: int) -> None:
        if self.bev_transform is None:
            return

        meters = self.bev_transform.meters_per_division
        pixels = int(round(meters * self.bev_transform.scale))
        pixels = max(40, pixels)

        for x in range(self.bev_transform.origin_x, right + pixels, pixels):
            if left <= x <= right:
                cv2.line(bev, (x, top), (x, bottom), self.grid_color, 1)
        for x in range(self.bev_transform.origin_x, left - pixels, -pixels):
            if left <= x <= right:
                cv2.line(bev, (x, top), (x, bottom), self.grid_color, 1)

        for y in range(self.bev_transform.origin_y, bottom + pixels, pixels):
            if top <= y <= bottom:
                cv2.line(bev, (left, y), (right, y), self.grid_color, 1)
        for y in range(self.bev_transform.origin_y, top - pixels, -pixels):
            if top <= y <= bottom:
                cv2.line(bev, (left, y), (right, y), self.grid_color, 1)

    def _draw_bev_axes(self, bev: np.ndarray, left: int, top: int, right: int, bottom: int) -> None:
        if self.bev_transform is None:
            return

        ox = self.bev_transform.origin_x
        oy = self.bev_transform.origin_y

        if top <= oy <= bottom:
            cv2.line(bev, (left, oy), (right, oy), self.axis_color, 1)
            cv2.putText(bev, "X →", (right - 40, max(top + 16, oy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.axis_color, 1, lineType=cv2.LINE_AA)

        if left <= ox <= right:
            cv2.line(bev, (ox, top), (ox, bottom), self.axis_color, 1)
            cv2.putText(bev, "Y ↑", (min(right - 34, ox + 6), top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.axis_color, 1, lineType=cv2.LINE_AA)

    def _draw_bev_scale(self, bev: np.ndarray) -> None:
        if self.bev_transform is None or self.layout is None:
            return

        meters = self.bev_transform.meters_per_division
        px = int(round(meters * self.bev_transform.scale))
        px = max(px, 40)

        x1 = self.layout.padding + 12
        y = self.layout.bev_height - self.layout.padding - 14
        x2 = x1 + px

        cv2.line(bev, (x1, y), (x2, y), self.color_text_fg, 2)
        cv2.line(bev, (x1, y - 5), (x1, y + 5), self.color_text_fg, 2)
        cv2.line(bev, (x2, y - 5), (x2, y + 5), self.color_text_fg, 2)
        cv2.putText(
            bev,
            f"{meters:.0f} m",
            (x1, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            self.color_text_fg,
            1,
            lineType=cv2.LINE_AA,
        )

    def _trajectory_to_bev_pixels(self, points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        pixels: List[Tuple[int, int]] = []
        for point in points:
            p = self._world_to_bev_pixel(point)
            if p is not None:
                pixels.append(p)
        return pixels

    def _world_to_bev_pixel(self, point: Tuple[float, float]) -> Tuple[int, int] | None:
        if self.bev_transform is None or self.layout is None:
            return None

        x, y = float(point[0]), float(point[1])

        px = int(round(self.bev_transform.origin_x + x * self.bev_transform.scale))
        py = int(round(self.bev_transform.origin_y - y * self.bev_transform.scale))

        left = self.layout.padding
        top = 56
        right = self.layout.bev_width - self.layout.padding
        bottom = self.layout.bev_height - self.layout.padding

        px = max(left, min(px, right))
        py = max(top, min(py, bottom))
        return (px, py)

    def _draw_polyline_with_points(
        self,
        canvas: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        point_radius: int,
        thickness: int,
    ) -> None:
        if len(points) >= 2:
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, color, thickness, lineType=cv2.LINE_AA)

        for x, y in points:
            cv2.circle(canvas, (x, y), point_radius, color, thickness=-1)

    def _draw_triangle_marker(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        color: Tuple[int, int, int],
    ) -> None:
        cx, cy = center
        pts = np.array(
            [
                [cx, cy - 10],
                [cx - 8, cy + 6],
                [cx + 8, cy + 6],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, pts, color)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], True, (255, 255, 255), 1)

    def _draw_velocity_arrow(self, bev: np.ndarray, obj: TrackedObject) -> None:
        if self.bev_transform is None:
            return

        vx, vy = obj.velocity
        mag = math.sqrt(vx * vx + vy * vy)
        if mag < 1e-6:
            return

        start = self._world_to_bev_pixel(
            (
                obj.anchor_world_position[0]
                + obj.world_position[0],
                obj.anchor_world_position[1]
                + obj.world_position[1],
            )
        )
        if start is None:
            return

        arrow_len = max(18, int(min(60, mag * self.bev_transform.scale * 0.6)))
        dx = int(round((vx / mag) * arrow_len))
        dy = int(round((-vy / mag) * arrow_len))

        end = (start[0] + dx, start[1] + dy)

        cv2.arrowedLine(
            bev,
            start,
            end,
            self.axis_color,
            2,
            line_type=cv2.LINE_AA,
            tipLength=0.28,
        )

    # =====================================================
    # Absolute Trajectory Helpers
    # =====================================================

    def _get_absolute_history(
        self,
        obj: TrackedObject,
    ) -> List[Tuple[float, float]]:
        ax, ay = obj.anchor_world_position
        return [
            (ax + x, ay + y)
            for x, y in obj.history
        ]

    def _get_absolute_prediction(
        self,
        obj: TrackedObject,
    ) -> List[Tuple[float, float]]:
        ax, ay = obj.anchor_world_position
        return [
            (ax + x, ay + y)
            for x, y in obj.future_prediction
        ]

    def _get_absolute_ground_truth(
        self,
        obj: TrackedObject,
    ) -> List[Tuple[float, float]]:
        ax, ay = obj.anchor_world_position
        return [
            (ax + x, ay + y)
            for x, y in obj.future_ground_truth
        ]

    # =====================================================
    # Numeric Helpers
    # =====================================================

    def _speed_magnitude(self, velocity: Tuple[float, float]) -> float:
        vx, vy = velocity
        return float(math.sqrt(vx ** 2 + vy ** 2))

    def _safe_mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    # =====================================================
    # Risk Helpers
    # =====================================================

    def _risk_color(self, score: float) -> Tuple[int, int, int]:
        if score >= config.HIGH_RISK_THRESHOLD:
            return self.color_risk_high
        if score >= self.risk_mid_threshold:
            return self.color_risk_mid
        if score >= self.risk_low_threshold:
            return self.color_risk_low
        return self.color_risk_low


# =========================================================
# Entry Point (optional)
# ==========================================================

def main() -> None:
    results_json = Path(config.PREDICTION_DIR) / "results.json"
    visualizer = ResultsVisualizer(results_path=results_json)
    visualizer.load_results()
    output_path = visualizer.render(output_name="result.mp4")
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
