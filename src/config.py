"""
=========================================================
Project : Temporal Risk Prediction for Autonomous Driving
Author  : Jayanth K
Version : 0.2.0
=========================================================

Global Configuration File

This file contains all configurable parameters used
throughout the project.

Every module should import values only from this file.
"""

# =========================================================
# PROJECT
# =========================================================

PROJECT_NAME = "Temporal Risk Prediction"

VERSION = "0.2.0"

RANDOM_SEED = 42

# Runtime device selection should be done via:
# torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda"

# Separate seed if you want dataset splits independent of other RNG use
DATASET_SPLIT_SEED = 42

# =========================================================
# DATASET SETTINGS
# =========================================================

DATASET_NAME = "Argoverse 2 Sensor Dataset"

# Root dataset directory
DATASET_DIR = "datasets/av2/sensor"

# Dataset splits (by log)
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

# Camera used throughout the project
CAMERA_NAME = "ring_front_center"

IMAGE_EXTENSION = ".jpg"

# Argoverse front camera resolution (approx.)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200

# Approximate frame rate for ring_front_center
FRAME_RATE = 20

# =========================================================
# DATASET BUILDER SETTINGS
# =========================================================

# Dynamic classes used for trajectory prediction
DATASET_TARGET_CLASSES = [
    "PEDESTRIAN",
    "REGULAR_VEHICLE",
    "BUS",
    "BOX_TRUCK",
    "LARGE_VEHICLE",
    "WHEELCHAIR",
]

# Ignore static objects
IGNORE_CLASSES = [
    "SIGN",
    "CONSTRUCTION_CONE",
]

# Timestamp matching
MATCH_NEAREST_TIMESTAMP = True

# Build sequences only from sufficiently long tracks
MIN_TRACK_POINTS = 20

# Output processed dataset (JSON for now)
PROCESSED_DATASET_DIR = "datasets/processed"

# =========================================================
# TEMPORAL SETTINGS
# =========================================================

# Number of previous frames (history)
SEQUENCE_LENGTH = 10

# Number of future frames (prediction horizon)
PREDICTION_HORIZON = 5

# Sliding window stride
SEQUENCE_STRIDE = 1

# Minimum history before prediction
MIN_SEQUENCE_LENGTH = 10

# =========================================================
# DETECTION SETTINGS
# =========================================================

DETECTOR = "YOLOv8"

YOLO_MODEL = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.25

IOU_THRESHOLD = 0.50

# YOLO class names
YOLO_TARGET_CLASSES = [
    "person",
    "car",
    "bus",
    "truck",
]

# =========================================================
# TRACKING SETTINGS
# =========================================================

TRACKER = "ByteTrack"

TRACK_BUFFER = 30

MAX_HISTORY = 30

MIN_TRACK_LENGTH = 10

# =========================================================
# FEATURE EXTRACTION
# =========================================================

FEATURE_EXTRACTOR = "ResNet18"
FEATURE_SIZE = 512

# Use ImageNet-pretrained weights for ResNet18
PRETRAINED = True

# Normalization for ResNet18 (ImageNet)
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

# =========================================================
# MODEL SETTINGS
# =========================================================

SUPPORTED_MODELS = [
    "LSTM",
    "BiLSTM",
    "GRU",
]

# Chosen temporal model for this experiment
MODEL_NAME = "LSTM"  # can switch to "LSTM" or "GRU" without code changes

# (x,y) input
INPUT_SIZE = 2

# Predicted (x,y)
OUTPUT_SIZE = 2

HIDDEN_SIZE = 128

NUM_LAYERS = 2

DROPOUT = 0.2

BIDIRECTIONAL = True  # used when MODEL_NAME == "BiLSTM"

# =========================================================
# TRAINING SETTINGS
# =========================================================

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

EPOCHS = 10

TRAIN_SPLIT = 0.70

VAL_SPLIT = 0.15

TEST_SPLIT = 0.15

NUM_WORKERS = 4

SHUFFLE_DATASET = True

# =========================================================
# RISK SETTINGS
# =========================================================

LOW_RISK_THRESHOLD = 0.30
MEDIUM_RISK_THRESHOLD = 0.60
HIGH_RISK_THRESHOLD = 0.80

# =========================================================
# VISUALIZATION SETTINGS
# =========================================================

# Font & line thickness
VIS_FONT_SCALE = 0.6
VIS_FONT_THICKNESS = 1

VIS_LINE_THICKNESS = 2
VIS_CIRCLE_RADIUS = 3

# Video FPS for results.mp4
VIS_VIDEO_FPS = 20

# Legend layout
VIS_LEGEND_MARGIN = 10
VIS_LEGEND_LINE_LENGTH = 30

# Trajectory colors (BGR)
VIS_COLOR_HISTORY = (255, 0, 0)        # Blue-ish for history
VIS_COLOR_GT = (0, 255, 0)            # Green for ground truth
VIS_COLOR_PRED = (0, 0, 255)          # Red for prediction

# Risk colors (bbox / label)
VIS_COLOR_RISK_LOW = (0, 255, 0)      # Green
VIS_COLOR_RISK_MID = (0, 165, 255)    # Orange
VIS_COLOR_RISK_HIGH = (0, 0, 255)     # Red

# Text colors
VIS_COLOR_TEXT_BG = (0, 0, 0)         # Black
VIS_COLOR_TEXT_FG = (255, 255, 255)   # White

# Optional toggles (still useful for visualizer/drawing)
BOX_THICKNESS = VIS_LINE_THICKNESS
TRAJECTORY_THICKNESS = VIS_LINE_THICKNESS
PREDICTION_THICKNESS = VIS_LINE_THICKNESS

SHOW_TRACK_ID = True
SHOW_TRAJECTORY = True
SHOW_PREDICTION = True
SHOW_FPS = True

# =========================================================
# OUTPUT PATHS
# =========================================================

MODEL_DIR = "models"

OUTPUT_DIR = "outputs"

OUTPUT_VIDEOS_DIR = "outputs/videos"

# Single-file path; main.py will write result.mp4 here
OUTPUT_VIDEO = "outputs/videos/result.mp4"

OUTPUT_CSV = "outputs/trajectories/trajectories.csv"

LOG_DIR = "outputs/logs"

FIGURE_DIR = "outputs/figures"

CHECKPOINT_DIR = "outputs/checkpoints"

PREDICTION_DIR = "outputs/predictions"
