"""
=========================================================
Project : Temporal Risk Prediction for Autonomous Driving
Author  : Jayanth K
Version : 1.0.0
=========================================================

Prediction Model

This module implements the temporal prediction network used
for future trajectory estimation.

Pipeline

RGB Image
    ↓
ResNet18 Feature Extraction
    ↓
Temporal Model (LSTM / BiLSTM)
    ↓
Future Trajectory Prediction
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image

import config


# =========================================================
# ResNet18 Feature Extractor
# =========================================================

class ResNet18FeatureExtractor(nn.Module):
    """
    Extract a feature vector from an RGB image using ResNet18.

    The final classification layer is removed so the network
    acts only as a feature extractor.
    """

    def __init__(self) -> None:
        super().__init__()

        # Configuration-driven backbone selection
        if config.FEATURE_EXTRACTOR != "ResNet18":
            raise ValueError(
                f"FEATURE_EXTRACTOR={config.FEATURE_EXTRACTOR} is not supported; "
                "only 'ResNet18' is implemented."
            )

        use_pretrained = bool(config.PRETRAINED)

        if use_pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
        else:
            backbone = models.resnet18(weights=None)

        # Remove final fully-connected layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Feature dimension after global average pooling
        self.feature_dim = config.FEATURE_SIZE  # typically 512

        # Standard ImageNet transforms from config
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.IMAGE_MEAN,
                std=config.IMAGE_STD,
            ),
        ])

    @torch.no_grad()
    def extract(
        self,
        image_path: str | Path,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Extract a feature vector from an image path.

        This is the typical path for offline training using
        TrainingSample.image_path.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(path)

        image = Image.open(path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)  # (1, C, H, W)

        if device is not None:
            image_tensor = image_tensor.to(device)
            self.backbone.to(device)

        feature = self.backbone(image_tensor)  # (1, 512, 1, 1)
        feature = feature.flatten(start_dim=1)  # (1, 512)

        return feature.squeeze(0)  # (512,)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using an already transformed image tensor.

        Expects image of shape (B, C, H, W) in normalized form.
        """
        feature = self.backbone(image)          # (B, 512, 1, 1)
        feature = feature.flatten(start_dim=1)  # (B, 512)
        return feature


# =========================================================
# LSTM / BiLSTM Predictors
# =========================================================

class LSTMPredictor(nn.Module):
    """
    Temporal predictor using a unidirectional LSTM.

    Input per timestep:
        [x, y, feature]  -> concatenated vector of size (2 + feature_dim)

    Output:
        sequence of future (x, y) positions.
    """

    def __init__(
        self,
        feature_dim: int,
        history_length: int,
        future_length: int,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.history_length = history_length
        self.future_length = future_length

        self.input_dim = 2 + feature_dim  # x, y, feature
        self.hidden_size = config.HIDDEN_SIZE
        self.num_layers = config.NUM_LAYERS
        self.dropout = config.DROPOUT

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        self.output_head = nn.Linear(self.hidden_size, 2 * self.future_length)

    def forward(
        self,
        history_xy: torch.Tensor,   # (B, H, 2)
        feature: torch.Tensor,      # (B, feature_dim)
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            history_xy: normalized history positions (B, H, 2)
            feature: image features (B, feature_dim)

        Returns:
            future_xy: predicted future positions (B, F, 2)
        """
        batch_size, H, _ = history_xy.shape
        if H != self.history_length:
            raise ValueError(
                f"Expected history length {self.history_length}, got {H}."
            )

        # Expand feature across timesteps: (B, H, feature_dim)
        feature_expanded = feature.unsqueeze(1).expand(-1, H, -1)

        # Concatenate along feature dimension: (B, H, 2 + feature_dim)
        lstm_input = torch.cat([history_xy, feature_expanded], dim=-1)

        lstm_output, _ = self.lstm(lstm_input)  # (B, H, hidden)
        # Use last hidden state across time
        last_hidden = lstm_output[:, -1, :]     # (B, hidden)

        future_flat = self.output_head(last_hidden)  # (B, 2F)
        future_xy = future_flat.view(batch_size, self.future_length, 2)
        return future_xy


class BiLSTMPredictor(nn.Module):
    """
    Temporal predictor using a bidirectional LSTM.

    Same inputs/outputs as LSTMPredictor, but bidirectional.
    """

    def __init__(
        self,
        feature_dim: int,
        history_length: int,
        future_length: int,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.history_length = history_length
        self.future_length = future_length

        self.input_dim = 2 + feature_dim
        self.hidden_size = config.HIDDEN_SIZE
        self.num_layers = config.NUM_LAYERS
        self.dropout = config.DROPOUT

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Bidirectional doubles hidden size
        self.output_head = nn.Linear(self.hidden_size * 2, 2 * self.future_length)

    def forward(
        self,
        history_xy: torch.Tensor,   # (B, H, 2)
        feature: torch.Tensor,      # (B, feature_dim)
    ) -> torch.Tensor:
        batch_size, H, _ = history_xy.shape
        if H != self.history_length:
            raise ValueError(
                f"Expected history length {self.history_length}, got {H}."
            )

        feature_expanded = feature.unsqueeze(1).expand(-1, H, -1)
        lstm_input = torch.cat([history_xy, feature_expanded], dim=-1)

        lstm_output, _ = self.lstm(lstm_input)      # (B, H, hidden*2)
        last_hidden = lstm_output[:, -1, :]         # (B, hidden*2)

        future_flat = self.output_head(last_hidden) # (B, 2F)
        future_xy = future_flat.view(batch_size, self.future_length, 2)
        return future_xy


# =========================================================
# Temporal Predictor Wrapper
# =========================================================

class TemporalPredictor(nn.Module):
    """
    High-level temporal prediction model.

    Combines:
        ResNet18FeatureExtractor
        LSTMPredictor or BiLSTMPredictor

    Provides:
        forward()  : batch training/inference
        predict()  : single-sample convenience
    """

    def __init__(self) -> None:
        super().__init__()

        # Sequence lengths from config (must match dataset_builder)
        history_length = config.SEQUENCE_LENGTH
        future_length = config.PREDICTION_HORIZON

        self.feature_extractor = ResNet18FeatureExtractor()
        feature_dim = self.feature_extractor.feature_dim

        # Model selection from config.MODEL_NAME
        model_name = config.MODEL_NAME.upper()
        if model_name == "BILSTM":
            self.temporal_model = BiLSTMPredictor(
                feature_dim=feature_dim,
                history_length=history_length,
                future_length=future_length,
            )
        elif model_name == "LSTM":
            self.temporal_model = LSTMPredictor(
                feature_dim=feature_dim,
                history_length=history_length,
                future_length=future_length,
            )
        else:
            raise ValueError(
                f"MODEL_NAME={config.MODEL_NAME} is not supported; "
                "supported: 'LSTM', 'BiLSTM'."
            )

        self.history_length = history_length
        self.future_length = future_length

    def forward(
        self,
        history_xy: torch.Tensor,   # (B, H, 2)
        image_tensor: torch.Tensor, # (B, C, H_img, W_img) normalized
    ) -> torch.Tensor:
        """
        Forward pass for batch inputs with in-memory image tensors.

        This is ideal for streaming inference where images are
        already loaded/normalized (e.g. from OpenCV).
        """
        features = self.feature_extractor(image_tensor)  # (B, 512)
        future_xy = self.temporal_model(history_xy, features)
        return future_xy

    @torch.no_grad()
    def predict(
        self,
        history_xy: torch.Tensor,   # (H, 2)
        image_path: str | Path,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """
        Convenience method for single-sample inference from disk.

        history_xy is a single sequence (H, 2), image_path is used
        to extract a feature via ResNet18FeatureExtractor.extract().
        """
        self.eval()

        if device is None:
            device = torch.device("cpu")

        history_xy = history_xy.unsqueeze(0).to(device)  # (1, H, 2)

        feature = self.feature_extractor.extract(image_path, device=device)
        feature = feature.unsqueeze(0)  # (1, 512)

        future_xy = self.temporal_model(history_xy, feature)  # (1, F, 2)
        return future_xy.squeeze(0)  # (F, 2)

    def save(self, path: str | Path) -> None:
        """
        Save model state_dict to disk.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path.as_posix())

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> None:
        """
        Load model state_dict from disk.
        """
        path = Path(path)
        state = torch.load(path.as_posix(), map_location=map_location)
        self.load_state_dict(state)
