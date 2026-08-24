"""
=========================================================
Project : Temporal Risk Prediction for Autonomous Driving
Author  : Jayanth K
Version : 1.0.0
=========================================================

Main Orchestration Script

This script does NOT implement any algorithms.
It only orchestrates the frozen modules:

    - config.py
    - predict.py          (TemporalPredictor)
    - train.py            (TrajectoryDataset)
    - evaluate.py         (metrics: ADE, FDE, RMSE)
    - risk.py             (risk estimation)
    - results_analyzer.py (ResultsVisualizer)

Pipeline (test-time):

processed_test.json
        ↓
TemporalPredictor (BiLSTM)
        ↓
ADE / FDE / RMSE
        ↓
Risk Score / Level
        ↓
results["frames"]
        ↓
results.json
        ↓
results.mp4
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple

import json
import torch
from tqdm import tqdm

import config
from predict import TemporalPredictor
from train import TrajectoryDataset, VAL_TRANSFORM
from evaluate import compute_ade, compute_fde, compute_rmse
from risk import evaluate_risk
from tracked_object import TrackedObject
#from results_analyzer_v3 import ResultsVisualizer


# =========================================================
# Initialization
# =========================================================

def initialize_model() -> TemporalPredictor:
    """
    Load the trained temporal prediction model from disk
    and move it to the appropriate device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalPredictor()
    model_path = Path(config.MODEL_DIR) / "lstm" / "best_lstm.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model checkpoint not found: {model_path}"
        )

    state = torch.load(model_path.as_posix(), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    print("Loading trained model...")
    print("✓ Model loaded")

    return model


def load_test_dataset() -> TrajectoryDataset:
    """
    Load the processed test dataset (processed_test.json)
    using the same TrajectoryDataset used during training.
    """
    data_root = Path(config.PROCESSED_DATASET_DIR)
    test_json = data_root / "processed_test.json"

    if not test_json.exists():
        raise FileNotFoundError(test_json)

    print("Loading processed test dataset...")

    dataset = TrajectoryDataset(
        json_path=test_json,
        transform=VAL_TRANSFORM,
    )

    print(f"✓ {len(dataset)} samples loaded")
    return dataset


# =========================================================
# Inference + Risk Evaluation
# =========================================================

def run_inference(
    model: TemporalPredictor,
    dataset: TrajectoryDataset,
) -> Dict[str, Any]:
    """
    Run prediction, evaluation, and risk estimation over the
    entire test dataset.

    Returns a dict with:
        "frames"  : list of grouped scene frame dicts
        "metrics" : avg ADE / FDE / RMSE (in-memory only)
    """
    device = next(model.parameters()).device

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    scene_results: Dict[Tuple[str, int], Dict[str, Any]] = {}

    total_ade = 0.0
    total_fde = 0.0
    total_rmse = 0.0
    num_samples = 0

    print("Running inference...")

    pbar = tqdm(loader)
    for sample_index, batch in enumerate(pbar):
        image = batch["image"].to(device)
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        category = batch["category"][0]

        with torch.no_grad():
            pred_future = model(history, image)

        ade = compute_ade(pred_future, future)
        fde = compute_fde(pred_future, future)
        rmse = compute_rmse(pred_future, future)

        total_ade += ade
        total_fde += fde
        total_rmse += rmse
        num_samples += 1

        history_xy = batch["history"][0].cpu().numpy().tolist()
        future_gt_xy = batch["future"][0].cpu().numpy().tolist()
        future_pred_xy = pred_future[0].cpu().numpy().tolist()

        sample = dataset.samples[sample_index]
        log_id = str(sample["log_id"])
        timestamp_ns = int(sample["timestamp_ns"])
        track_uuid = str(sample["track_uuid"])
        image_path = sample["image_path"]
        anchor_world_position = sample["anchor_world_position"]

        obj = TrackedObject(
            track_id=sample_index,
            track_uuid=track_uuid,
            class_name=str(category),
            confidence=1.0,
            bbox=(0, 0, 0, 0),
            center=(0.0, 0.0),
        )

        obj.history = [tuple(p) for p in history_xy]
        obj.future_ground_truth = [tuple(p) for p in future_gt_xy]
        obj.future_prediction = [tuple(p) for p in future_pred_xy]

        if future_pred_xy:
            obj.world_position = tuple(future_pred_xy[0])
        else:
            obj.world_position = (0.0, 0.0)

        obj.calculate_velocity()
        evaluate_risk(obj)

        obj.ade = float(ade)
        obj.fde = float(fde)

        track_dict = {
            "track_id": obj.track_id,
            "track_uuid": obj.track_uuid,
            "anchor_world_position": list(anchor_world_position),
            "class_name": obj.class_name,
            "confidence": obj.confidence,
            "bbox": list(obj.bbox),
            "center": list(obj.center),
            "world_position": list(obj.world_position),
            "velocity": list(obj.velocity),
            "risk_score": obj.risk_score,
            "risk_level": obj.risk_level,
            "ttc": obj.ttc,
            "history": [list(p) for p in obj.history],
            "future_prediction": [list(p) for p in obj.future_prediction],
            "future_ground_truth": [list(p) for p in obj.future_ground_truth],
            "ade": obj.ade,
            "fde": obj.fde,
        }

        scene_key = (log_id, timestamp_ns)

        if scene_key not in scene_results:
            scene_results[scene_key] = {
                "log_id": log_id,
                "timestamp_ns": timestamp_ns,
                "image_path": image_path,
                "tracks": [],
            }

        scene_results[scene_key]["tracks"].append(track_dict)

    frames: List[Dict[str, Any]] = []

    for frame_index, scene_key in enumerate(sorted(scene_results.keys())):
        scene = scene_results[scene_key]
        frames.append({
            "frame_index": frame_index,
            "log_id": scene["log_id"],
            "timestamp_ns": scene["timestamp_ns"],
            "image_path": scene["image_path"],
            "tracks": scene["tracks"],
        })

    print(f"Scenes: {len(frames)}")

    total_tracks = sum(len(f["tracks"]) for f in frames)
    print(f"Objects: {total_tracks}")

    if frames:
        print(
            f"Average objects/scene: "
            f"{total_tracks / len(frames):.2f}"
        )
        print(
            f"Maximum objects in one scene: "
            f"{max(len(f['tracks']) for f in frames)}"
        )
    else:
        print("Average objects/scene: 0.00")
        print("Maximum objects in one scene: 0")

    avg_ade = total_ade / max(num_samples, 1)
    avg_fde = total_fde / max(num_samples, 1)
    avg_rmse = total_rmse / max(num_samples, 1)

    return {
        "frames": frames,
        "metrics": {
            "avg_ade": avg_ade,
            "avg_fde": avg_fde,
            "avg_rmse": avg_rmse,
        },
    }


# =========================================================
# Save Results (frames only)
# =========================================================

def save_results_json(results: Dict[str, Any]) -> Path:
    """
    Save only results['frames'] to outputs/predictions/results.json.

    Metrics stay in-memory for print_summary().
    """
    output_dir = Path(config.PREDICTION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "results_lstm.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results["frames"], f, ensure_ascii=False, indent=2)

    print("Saving results...")
    print(f"✓ {out_path}")
    return out_path


# =========================================================
# Render Video
# =========================================================

def render_video(results_json_path: Path) -> Path:
    """
    Use ResultsVisualizer to turn results.json into result.mp4.
    """
    print("Rendering video...")

    visualizer = ResultsVisualizer(results_path=results_json_path)
    visualizer.load_results()
    video_path = visualizer.render(output_name="result.mp4")

    print(f"✓ {video_path}")
    return video_path


# =========================================================
# Print Summary
# =========================================================

def print_summary(results: Dict[str, Any]) -> None:
    """
    Print average ADE / FDE / RMSE to console.
    """
    metrics = results["metrics"]
    avg_ade = metrics["avg_ade"]
    avg_fde = metrics["avg_fde"]
    avg_rmse = metrics["avg_rmse"]

    print()
    print(f"Average ADE : {avg_ade:.2f} m")
    print(f"Average FDE : {avg_fde:.2f} m")
    print(f"Average RMSE: {avg_rmse:.2f} m")
    print()
    print("Done.")


# =========================================================
# Entry Point
# =========================================================

def main() -> None:
    predictor = initialize_model()
    dataset = load_test_dataset()

    results = run_inference(
        model=predictor,
        dataset=dataset,
    )

    results_json_path = save_results_json(results)
    render_video(results_json_path)
    print_summary(results)


if __name__ == "__main__":
    main()
