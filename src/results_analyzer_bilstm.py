"""
Results Analyzer / Visualizer V5
--------------------------------
Built strictly around the current results.json contract.

Important data contract:
- anchor_world_position = current object position in AV2 ego-reference coordinates.
- history/future_prediction/future_ground_truth = RELATIVE offsets from
  the last history point (which is [0, 0]).
- world_position = first predicted relative point used by risk.py; it is
  NOT the object's current scene position.
- bbox/center in the current results.json are placeholders [0,0,0,0]
  and [0,0], so V5 NEVER invents camera boxes or projects fake paths.

Therefore:
    absolute trajectory = anchor_world_position + relative trajectory

BEV:
    uses AV2 ego-reference coordinates directly.
    x = longitudinal/forward axis
    y = lateral axis
    ego = (0,0)
    object current position = anchor_world_position

This visualizer does not change inference, prediction, risk, ADE, FDE,
or results.json.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

import config
from tracked_object import TrackedObject


Point = Tuple[float, float]
Pixel = Tuple[int, int]
Color = Tuple[int, int, int]


@dataclass
class FrameResult:
    frame_index: int
    image_path: str
    log_id: str
    timestamp_ns: int
    tracks: List[TrackedObject]


class ResultsVisualizer:
    def __init__(
        self,
        results_path: str | Path,
        output_video_dir: str | Path | None = None,
    ) -> None:
        self.results_path = Path(results_path)
        self.output_video_dir = Path(
            output_video_dir
            if output_video_dir is not None
            else config.OUTPUT_VIDEOS_DIR
        )

        self.W = 1920
        self.H = 1080
        self.fps = getattr(config, "VIS_VIDEO_FPS", 10)

        # Dashboard
        self.margin = 24
        self.gap = 16

        # BEV: intentionally close to ego.
        # More forward than rearward because this is driving-oriented.
        self.forward_m = 12.0
        self.rear_m = 6.0
        self.left_m = 8.0
        self.right_m = 8.0

        # Colors (BGR)
        self.bg = (13, 15, 19)
        self.panel = (25, 29, 35)
        self.panel_alt = (31, 36, 43)
        self.grid = (58, 64, 73)
        self.axis = (145, 153, 164)

        self.white = (245, 246, 248)
        self.text = (225, 229, 235)
        self.muted = (165, 174, 186)
        self.faint = (112, 121, 133)

        self.history_color = (180, 185, 192)
        self.pred_color = (20, 230, 255)       # bright yellow in BGR
        self.gt_color = (70, 235, 105)         # green
        self.error_color = (245, 245, 245)

        # Risk: red / yellow / green
        self.risk_high = (45, 55, 255)
        self.risk_medium = (0, 215, 255)
        self.risk_low = (60, 210, 85)

        self.velocity_color = (205, 105, 245)
        self.ego_color = (235, 238, 242)

        self.frames: List[FrameResult] = []

    # =========================================================
    # Load
    # =========================================================

    def load_results(self) -> None:
        if not self.results_path.exists():
            raise FileNotFoundError(self.results_path)

        with self.results_path.open("r", encoding="utf-8") as f:
            raw_frames = json.load(f)

        self.frames = []

        for raw in raw_frames:
            tracks: List[TrackedObject] = []

            for t in raw.get("tracks", []):
                obj = TrackedObject(
                    track_id=int(t.get("track_id", 0)),
                    track_uuid=str(t.get("track_uuid", "")),
                    class_name=str(t.get("class_name", "UNKNOWN")),
                    confidence=float(t.get("confidence", 0.0)),
                    bbox=tuple(t.get("bbox", (0, 0, 0, 0))),
                    center=tuple(t.get("center", (0.0, 0.0))),
                    world_position=tuple(
                        t.get("world_position", (0.0, 0.0))
                    ),
                    anchor_world_position=tuple(
                        t.get("anchor_world_position", (0.0, 0.0))
                    ),
                    velocity=tuple(t.get("velocity", (0.0, 0.0))),
                )

                obj.history = [
                    tuple(p) for p in t.get("history", [])
                ]
                obj.future_prediction = [
                    tuple(p) for p in t.get("future_prediction", [])
                ]
                obj.future_ground_truth = [
                    tuple(p) for p in t.get("future_ground_truth", [])
                ]

                obj.risk_score = float(t.get("risk_score", 0.0))
                obj.risk_level = str(t.get("risk_level", "LOW"))
                obj.ttc = float(t.get("ttc", float("inf")))
                obj.ade = float(t.get("ade", 0.0))
                obj.fde = float(t.get("fde", 0.0))

                tracks.append(obj)

            self.frames.append(
                FrameResult(
                    frame_index=int(raw.get("frame_index", 0)),
                    image_path=str(raw.get("image_path", "")),
                    log_id=str(raw.get("log_id", "")),
                    timestamp_ns=int(raw.get("timestamp_ns", 0)),
                    tracks=tracks,
                )
            )

    # =========================================================
    # Main render
    # =========================================================

    def render(self, output_name: str = "result_v5_bilstm_direction_fixed.mp4") -> Path:
        if not self.frames:
            raise RuntimeError("No results loaded.")

        self.output_video_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_video_dir / output_name

        writer = cv2.VideoWriter(
            str(out),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.W, self.H),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {out}")

        for frame in self.frames:
            writer.write(self.render_frame(frame))

        writer.release()
        return out

    def render_frame(self, frame: FrameResult) -> np.ndarray:
        canvas = np.full(
            (self.H, self.W, 3),
            self.bg,
            dtype=np.uint8
        )

        header_h = 72
        pipeline_h = 42
        footer_h = 50

        y = self.margin

        header = self.draw_header(frame, self.W - 2 * self.margin, header_h)
        self.paste(canvas, header, self.margin, y)
        y += header_h + self.gap

        pipeline = self.draw_pipeline(self.W - 2 * self.margin, pipeline_h)
        self.paste(canvas, pipeline, self.margin, y)
        y += pipeline_h + self.gap

        available_h = (
            self.H
            - y
            - footer_h
            - self.margin
            - self.gap
        )

        top_h = int(available_h * 0.69)
        bottom_h = available_h - top_h

        camera_w = 650
        bev_w = self.W - 2 * self.margin - camera_w - self.gap

        camera = self.draw_camera(frame, camera_w, top_h)
        bev = self.draw_bev(frame, bev_w, top_h)

        self.paste(canvas, camera, self.margin, y)
        self.paste(
            canvas,
            bev,
            self.margin + camera_w + self.gap,
            y
        )

        y += top_h + self.gap

        summary_w = 650
        metrics_w = self.W - 2 * self.margin - summary_w - self.gap

        summary = self.draw_summary(frame, summary_w, bottom_h)
        metrics = self.draw_metrics(frame, metrics_w, bottom_h)

        self.paste(canvas, summary, self.margin, y)
        self.paste(
            canvas,
            metrics,
            self.margin + summary_w + self.gap,
            y
        )

        footer = self.draw_footer(
            frame,
            self.W - 2 * self.margin,
            footer_h
        )

        self.paste(
            canvas,
            footer,
            self.margin,
            self.H - self.margin - footer_h
        )

        return canvas

    # =========================================================
    # Header / pipeline
    # =========================================================

    def draw_header(self, frame: FrameResult, w: int, h: int) -> np.ndarray:
        p = self.blank(w, h, self.panel)
        self.border(p)

        cv2.putText(
            p,
            "Learning-Based Temporal Motion Prediction for Risk-Aware Autonomous Driving",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            self.white,
            2,
            cv2.LINE_AA,
        )

        model = "BiLSTM"
        dataset_name = getattr(
            config,
            "DATASET_NAME",
            "Argoverse 2 Sensor Dataset"
        )

        cv2.putText(
            p,
            f"RGB camera sequence  |  {model}  |  {dataset_name}",
            (18, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            self.muted,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            p,
            f"FRAME {frame.frame_index:06d}",
            (w - 170, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            self.white,
            1,
            cv2.LINE_AA,
        )


        # Presenter name — no group/team label.
        name = "Jayanth Narayana K"
        (name_w, _), _ = cv2.getTextSize(
            name,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            1,
        )
        cv2.putText(
            p,
            name,
            (w - name_w - 18, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            self.white,
            1,
            cv2.LINE_AA,
        )

        return p

    def draw_pipeline(self, w: int, h: int) -> np.ndarray:
        p = self.blank(w, h, self.panel_alt)
        self.border(p)

        stages = [
            "Camera",
            "History",
            "BiLSTM",
            "Future Prediction",
            "ADE / FDE",
            "Risk",
        ]

        x = 18
        y = h // 2

        for i, stage in enumerate(stages):
            cv2.circle(
                p, (x, y), 4,
                self.pred_color,
                -1,
                cv2.LINE_AA
            )

            cv2.putText(
                p,
                stage,
                (x + 10, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                self.text,
                1,
                cv2.LINE_AA,
            )

            tw = cv2.getTextSize(
                stage,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                1
            )[0][0]

            x += tw + 42

            if i < len(stages) - 1:
                cv2.arrowedLine(
                    p,
                    (x - 28, y),
                    (x, y),
                    self.muted,
                    1,
                    cv2.LINE_AA,
                    tipLength=0.18
                )

        return p

    # =========================================================
    # Camera
    # =========================================================

    def draw_camera(
        self,
        frame: FrameResult,
        w: int,
        h: int,
    ) -> np.ndarray:

        img = cv2.imread(frame.image_path)

        if img is None:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                img,
                "CAMERA IMAGE NOT FOUND",
                (25, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.white,
                2,
                cv2.LINE_AA
            )
            return img

        img = cv2.resize(img, (w, h))

        overlay = img.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (w, 82),
            (0, 0, 0),
            -1
        )
        img = cv2.addWeighted(
            overlay, 0.58,
            img, 0.42,
            0
        )

        cv2.putText(
            img,
            "CAMERA VIEW",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            self.white,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            "Current RGB frame",
            (16, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            self.muted,
            1,
            cv2.LINE_AA
        )

        # IMPORTANT:
        # Current results.json has bbox=[0,0,0,0].
        # Do not draw fake boxes.
        cv2.putText(
            img,
            "2D boxes unavailable in results.json",
            (16, 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            self.faint,
            1,
            cv2.LINE_AA
        )

        self.border(img)
        return img

    # =========================================================
    # Correct BEV
    # =========================================================

    def draw_bev(
        self,
        frame: FrameResult,
        w: int,
        h: int,
    ) -> np.ndarray:

        p = self.blank(w, h, self.panel)

        self.border(p)

        cv2.putText(
            p,
            "BIRD'S-EYE VIEW",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            self.white,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            p,
            "AV2 ego-reference coordinates  |  current object = anchor",
            (18, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            self.muted,
            1,
            cv2.LINE_AA
        )

        left = 18
        right = w - 18
        top = 68
        bottom = h - 18

        cv2.rectangle(
            p,
            (left, top),
            (right, bottom),
            self.panel_alt,
            -1
        )

        # Ego-relative plot:
        # x positive is forward, x negative is behind.
        # y positive is left, y negative is right in AV2 ego coordinates.
        def to_px(q: Point) -> Pixel:
            x, y = q

            # AV2 ego frame: +Y = LEFT.
            # Image/screen X increases to the RIGHT, so +Y must map
            # toward smaller screen X values.
            px = left + (
                (self.left_m - y)
                / (self.left_m + self.right_m)
            ) * (right - left)

            py = top + (
                (self.forward_m - x)
                / (self.forward_m + self.rear_m)
            ) * (bottom - top)

            return (
                int(np.clip(px, left, right)),
                int(np.clip(py, top, bottom))
            )

        # grid
        for x_m in np.arange(
            -self.rear_m,
            self.forward_m + 0.1,
            2.0
        ):
            a = to_px((x_m, -self.right_m))
            b = to_px((x_m, self.left_m))
            cv2.line(p, a, b, self.grid, 1, cv2.LINE_AA)

            if abs(x_m) > 0.01:
                cv2.putText(
                    p,
                    f"{x_m:+.0f}m",
                    (a[0] + 4, a[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    self.faint,
                    1,
                    cv2.LINE_AA
                )

        for y_m in np.arange(
            -self.right_m,
            self.left_m + 0.1,
            2.0
        ):
            a = to_px((-self.rear_m, y_m))
            b = to_px((self.forward_m, y_m))
            cv2.line(p, a, b, self.grid, 1, cv2.LINE_AA)

        ego = to_px((0.0, 0.0))

        cv2.line(
            p,
            (ego[0], top),
            (ego[0], bottom),
            self.axis,
            1,
            cv2.LINE_AA
        )

        cv2.line(
            p,
            (left, ego[1]),
            (right, ego[1]),
            self.axis,
            1,
            cv2.LINE_AA
        )

        # Forward arrow
        cv2.arrowedLine(
            p,
            (ego[0], ego[1] + 42),
            (ego[0], ego[1] - 70),
            self.white,
            2,
            cv2.LINE_AA,
            tipLength=0.16
        )

        cv2.putText(
            p,
            "FORWARD",
            (ego[0] + 10, top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            self.white,
            1,
            cv2.LINE_AA
        )

        self.draw_ego(p, ego)

        # Show only objects whose CURRENT anchor lies in the useful view.
        # This avoids 40–60 m-away tracks crushing the presentation.
        visible = []

        for obj in frame.tracks:
            ax, ay = obj.anchor_world_position

            if (
                -self.rear_m <= ax <= self.forward_m
                and
                -self.right_m <= ay <= self.left_m
            ):
                visible.append(obj)

        # Prioritize risk, then distance.
        visible.sort(
            key=lambda o: (
                o.risk_score,
                -self.distance_from_ego(o)
            ),
            reverse=True
        )

        max_objects = 14

        for obj in visible[:max_objects]:
            self.draw_bev_object(
                p,
                obj,
                to_px
            )

        # Header status
        cv2.putText(
            p,
            f"{len(visible)} nearby / {len(frame.tracks)} total tracks",
            (w - 235, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            self.muted,
            1,
            cv2.LINE_AA
        )

        # Legend
        self.draw_bev_legend(p, left + 14, bottom - 14)

        return p

    def draw_bev_object(
        self,
        p: np.ndarray,
        obj: TrackedObject,
        to_px,
    ) -> None:

        current = self.absolute_current(obj)

        history = [
            to_px(q)
            for q in self.absolute_history(obj)
        ]

        prediction = [
            to_px(q)
            for q in self.absolute_prediction(obj)
        ]

        ground_truth = [
            to_px(q)
            for q in self.absolute_ground_truth(obj)
        ]

        current_px = to_px(current)
        risk = self.risk_color(obj.risk_score)

        # Risk halo
        radius = int(
            14 + 22 * float(np.clip(obj.risk_score, 0, 1))
        )

        overlay = p.copy()

        cv2.circle(
            overlay,
            current_px,
            radius,
            risk,
            2,
            cv2.LINE_AA
        )

        p[:, :] = cv2.addWeighted(
            overlay,
            0.72,
            p,
            0.28,
            0
        )

        # History: thin gray solid
        self.polyline(
            p,
            history,
            self.history_color,
            2,
            markers=True,
            marker_radius=3
        )

        # Ground truth: green solid
        self.polyline(
            p,
            ground_truth,
            self.gt_color,
            3,
            markers=True,
            marker_radius=4
        )

        # Prediction: yellow dashed
        self.dashed_polyline(
            p,
            prediction,
            self.pred_color,
            4
        )

        # Current position
        cv2.circle(
            p,
            current_px,
            7,
            self.white,
            -1,
            cv2.LINE_AA
        )

        cv2.circle(
            p,
            current_px,
            10,
            risk,
            2,
            cv2.LINE_AA
        )

        # Prediction final point / FDE
        if prediction:
            final_pred = prediction[-1]

            cv2.line(
                p,
                final_pred,
                ground_truth[-1] if ground_truth else final_pred,
                self.error_color,
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                p,
                f"FDE {obj.fde:.2f}m",
                (
                    min(p.shape[1] - 90, final_pred[0] + 7),
                    max(75, final_pred[1] - 7)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                self.white,
                1,
                cv2.LINE_AA
            )

        # Label
        label = self.short_label(obj)

        label_x = current_px[0] + 13
        label_y = current_px[1] - 13

        if label_x > p.shape[1] - 80:
            label_x = current_px[0] - 65

        if label_y < 75:
            label_y = current_px[1] + 28

        cv2.putText(
            p,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.white,
            1,
            cv2.LINE_AA
        )

        # Risk label
        cv2.putText(
            p,
            obj.risk_level.upper(),
            (label_x, label_y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            risk,
            1,
            cv2.LINE_AA
        )

    # =========================================================
    # Summary
    # =========================================================

    def draw_summary(
        self,
        frame: FrameResult,
        w: int,
        h: int,
    ) -> np.ndarray:

        p = self.blank(w, h, self.panel)
        self.border(p)

        cv2.putText(
            p,
            "FRAME SUMMARY",
            (16, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            self.white,
            2,
            cv2.LINE_AA
        )

        tracks = frame.tracks

        avg_ade = self.mean([o.ade for o in tracks])
        avg_fde = self.mean([o.fde for o in tracks])
        avg_risk = self.mean([o.risk_score for o in tracks])

        low = sum(
            1 for o in tracks
            if o.risk_level.upper() == "LOW"
        )
        med = sum(
            1 for o in tracks
            if o.risk_level.upper() == "MEDIUM"
        )
        high = sum(
            1 for o in tracks
            if o.risk_level.upper() == "HIGH"
        )

        self.summary_item(
            p, 18, 59, "TRACKS", str(len(tracks))
        )
        self.summary_item(
            p, 18, 88, "AVG ADE", f"{avg_ade:.3f} m"
        )
        self.summary_item(
            p, 18, 117, "AVG FDE", f"{avg_fde:.3f} m"
        )

        self.summary_item(
            p, 250, 59, "AVG RISK", f"{avg_risk:.3f}"
        )
        self.summary_item(
            p, 250, 88, "LOW", str(low), self.risk_low
        )
        self.summary_item(
            p, 250, 117, "MEDIUM", str(med), self.risk_medium
        )
        self.summary_item(
            p, 250, 146, "HIGH", str(high), self.risk_high
        )

        cv2.putText(
            p,
            "History: 10 frames",
            (18, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            self.muted,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            p,
            "Prediction horizon: 5 future points",
            (18, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            self.muted,
            1,
            cv2.LINE_AA
        )

        return p

    # =========================================================
    # Metric cards
    # =========================================================

    def draw_metrics(
        self,
        frame: FrameResult,
        w: int,
        h: int,
    ) -> np.ndarray:

        p = self.blank(w, h, self.panel)
        self.border(p)

        cv2.putText(
            p,
            "PRIORITY OBJECTS",
            (16, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            self.white,
            2,
            cv2.LINE_AA
        )

        tracks = sorted(
            frame.tracks,
            key=lambda o: (
                o.risk_score,
                o.ttc if math.isfinite(o.ttc) else 999.0
            ),
            reverse=True
        )

        n = min(4, len(tracks))

        if n == 0:
            return p

        gap = 10
        card_w = (w - 28 - gap * (n - 1)) // n
        card_h = h - 48

        for i, obj in enumerate(tracks[:n]):

            x = 14 + i * (card_w + gap)
            y = 40

            risk = self.risk_color(obj.risk_score)

            cv2.rectangle(
                p,
                (x, y),
                (x + card_w, y + card_h),
                self.panel_alt,
                -1
            )

            cv2.rectangle(
                p,
                (x, y),
                (x + 5, y + card_h),
                risk,
                -1
            )

            cv2.putText(
                p,
                self.short_label(obj),
                (x + 14, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                self.white,
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                p,
                obj.class_name.upper(),
                (x + 14, y + 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                self.muted,
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                p,
                obj.risk_level.upper(),
                (x + 14, y + 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                risk,
                1,
                cv2.LINE_AA
            )

            values = [
                ("Risk", f"{obj.risk_score:.2f}"),
                ("ADE", f"{obj.ade:.3f} m"),
                ("FDE", f"{obj.fde:.3f} m"),
                (
                    "TTC",
                    "inf"
                    if math.isinf(obj.ttc)
                    else f"{obj.ttc:.2f} s"
                ),
                (
                    "Speed",
                    f"{math.hypot(*obj.velocity):.2f} m/s"
                ),
            ]

            yy = y + 88

            for label, value in values:
                cv2.putText(
                    p,
                    label,
                    (x + 14, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.33,
                    self.muted,
                    1,
                    cv2.LINE_AA
                )

                cv2.putText(
                    p,
                    value,
                    (x + 73, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    self.white,
                    1,
                    cv2.LINE_AA
                )

                yy += 21

        return p

    # =========================================================
    # Footer
    # =========================================================

    def draw_footer(
        self,
        frame: FrameResult,
        w: int,
        h: int,
    ) -> np.ndarray:

        p = self.blank(w, h, self.panel_alt)
        self.border(p)

        entries = [
            ("History", self.history_color),
            ("Prediction", self.pred_color),
            ("Ground Truth", self.gt_color),
            ("Risk", self.risk_high),
        ]

        x = 18

        for label, color in entries:

            cv2.line(
                p,
                (x, 25),
                (x + 28, 25),
                color,
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                p,
                label,
                (x + 38, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                self.white,
                1,
                cv2.LINE_AA
            )

            x += 145

        cv2.putText(
            p,
            "Relative trajectories + anchor  |  Prediction vs GT  |  Risk-aware evaluation",
            (w - 650, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            self.muted,
            1,
            cv2.LINE_AA
        )

        return p

    # =========================================================
    # Geometry
    # =========================================================

    def absolute_current(self, obj: TrackedObject) -> Point:
        """
        Current object position in the AV2 ego-reference frame.

        anchor_world_position is the absolute anchor position.
        """
        ax, ay = obj.anchor_world_position
        return float(ax), float(ay)

    def absolute_history(self, obj: TrackedObject) -> List[Point]:
        ax, ay = obj.anchor_world_position
        return [
            (float(ax + x), float(ay + y))
            for x, y in obj.history
        ]

    def absolute_prediction(self, obj: TrackedObject) -> List[Point]:
        ax, ay = obj.anchor_world_position
        return [
            (float(ax + x), float(ay + y))
            for x, y in obj.future_prediction
        ]

    def absolute_ground_truth(self, obj: TrackedObject) -> List[Point]:
        ax, ay = obj.anchor_world_position
        return [
            (float(ax + x), float(ay + y))
            for x, y in obj.future_ground_truth
        ]

    # =========================================================
    # Drawing helpers
    # =========================================================

    def draw_ego(self, p: np.ndarray, ego: Pixel) -> None:

        cx, cy = ego

        # vehicle footprint
        cv2.rectangle(
            p,
            (cx - 13, cy - 27),
            (cx + 13, cy + 27),
            self.ego_color,
            -1
        )

        cv2.rectangle(
            p,
            (cx - 13, cy - 27),
            (cx + 13, cy + 27),
            self.white,
            2,
            cv2.LINE_AA
        )

        # heading
        pts = np.array([
            [cx, cy - 45],
            [cx - 9, cy - 27],
            [cx + 9, cy - 27]
        ], dtype=np.int32)

        cv2.fillConvexPoly(
            p,
            pts,
            self.white
        )

        cv2.putText(
            p,
            "EGO",
            (cx - 18, cy + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            self.white,
            1,
            cv2.LINE_AA
        )

    def polyline(
        self,
        p: np.ndarray,
        points: List[Pixel],
        color: Color,
        thickness: int,
        markers: bool = False,
        marker_radius: int = 3,
    ) -> None:

        if len(points) < 2:
            return

        arr = np.asarray(
            points,
            dtype=np.int32
        ).reshape(-1, 1, 2)

        cv2.polylines(
            p,
            [arr],
            False,
            color,
            thickness,
            cv2.LINE_AA
        )

        if markers:
            for q in points:
                cv2.circle(
                    p,
                    q,
                    marker_radius,
                    color,
                    -1,
                    cv2.LINE_AA
                )

    def dashed_polyline(
        self,
        p: np.ndarray,
        points: List[Pixel],
        color: Color,
        thickness: int,
    ) -> None:

        if len(points) < 2:
            return

        dash = 10.0
        gap = 7.0

        for a, b in zip(points[:-1], points[1:]):

            x1, y1 = a
            x2, y2 = b

            dx = x2 - x1
            dy = y2 - y1

            distance = math.hypot(dx, dy)

            if distance <= 0:
                continue

            ux = dx / distance
            uy = dy / distance

            s = 0.0

            while s < distance:

                e = min(
                    s + dash,
                    distance
                )

                p1 = (
                    int(x1 + ux * s),
                    int(y1 + uy * s)
                )

                p2 = (
                    int(x1 + ux * e),
                    int(y1 + uy * e)
                )

                cv2.line(
                    p,
                    p1,
                    p2,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

                s += dash + gap

        for q in points:
            cv2.circle(
                p,
                q,
                4,
                color,
                -1,
                cv2.LINE_AA
            )

    def draw_bev_legend(
        self,
        p: np.ndarray,
        x: int,
        y: int,
    ) -> None:

        entries = [
            ("History", self.history_color),
            ("Prediction", self.pred_color),
            ("Ground Truth", self.gt_color),
        ]

        for label, color in entries:

            cv2.line(
                p,
                (x, y),
                (x + 24, y),
                color,
                3,
                cv2.LINE_AA
            )

            cv2.putText(
                p,
                label,
                (x + 30, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                self.white,
                1,
                cv2.LINE_AA
            )

            x += 120

    def summary_item(
        self,
        p: np.ndarray,
        x: int,
        y: int,
        label: str,
        value: str,
        color: Optional[Color] = None,
    ) -> None:

        cv2.putText(
            p,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            self.muted,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            p,
            value,
            (x + 105, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color if color is not None else self.white,
            1,
            cv2.LINE_AA
        )

    def short_label(self, obj: TrackedObject) -> str:

        name = obj.class_name.upper()

        if "PED" in name:
            prefix = "P"

        elif (
            "VEH" in name
            or "CAR" in name
            or "TRUCK" in name
            or "BUS" in name
        ):
            prefix = "V"

        elif (
            "CYC" in name
            or "BICYCLE" in name
        ):
            prefix = "C"

        else:
            prefix = "O"

        return f"{prefix}{obj.track_id}"

    def distance_from_ego(
        self,
        obj: TrackedObject,
    ) -> float:

        x, y = obj.anchor_world_position
        return math.hypot(x, y)

    def risk_color(self, score: float) -> Color:

        high = getattr(
            config,
            "HIGH_RISK_THRESHOLD",
            None
        )

        medium = getattr(
            config,
            "MEDIUM_RISK_THRESHOLD",
            None
        )

        if high is not None and score >= high:
            return self.risk_high

        if medium is not None and score >= medium:
            return self.risk_medium

        # Fallback to the serialized semantic label is not available
        # here, so use score thresholds only.
        if score >= 0.70:
            return self.risk_high

        if score >= 0.40:
            return self.risk_medium

        return self.risk_low

    def mean(self, values: List[float]) -> float:

        return (
            float(sum(values) / len(values))
            if values
            else 0.0
        )

    def blank(
        self,
        w: int,
        h: int,
        color: Color,
    ) -> np.ndarray:

        return np.full(
            (h, w, 3),
            color,
            dtype=np.uint8
        )

    def border(self, p: np.ndarray) -> None:

        cv2.rectangle(
            p,
            (0, 0),
            (p.shape[1] - 1, p.shape[0] - 1),
            (72, 80, 91),
            1
        )

    def paste(
        self,
        canvas: np.ndarray,
        panel: np.ndarray,
        x: int,
        y: int,
    ) -> None:

        h, w = panel.shape[:2]

        canvas[
            y:y + h,
            x:x + w
        ] = panel


def main() -> None:

    results_path = (
        Path(config.PREDICTION_DIR)
        / "results.json"
    )

    visualizer = ResultsVisualizer(
        results_path=results_path
    )

    visualizer.load_results()

    output = visualizer.render(
        output_name="result_bilstm.mp4"
    )

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

