#!/usr/bin/env python3
"""
mcp_ensemble_smoke_test.py

Local smoke test:
- loads 6 member models from *_state_dict.pt
- runs inference on test images
- prints per-image predictions

Usage:
  python mcp_ensemble_smoke_test.py
  python mcp_ensemble_smoke_test.py --limit 25
  python mcp_ensemble_smoke_test.py --image /path/to/file.jpg --image /path/to/file2.jpg
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path("/Users/naraptis/Desktop/mcp")

DATA_ROOT = PROJECT_ROOT / "blood_cancer_cells_split"
TEST_ROOT = DATA_ROOT / "te"

NETWORK_SIZE = 224
NETWORK_POST_AUGMENTATION_INSET = 10  # crop inset used at test-time in your notebook
SQUEEZE_RATIO = 32

ALLOWED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
_ALLOWED_IMAGE_EXTENSION_SET = {ext.lower().lstrip(".") for ext in ALLOWED_IMAGE_EXTENSIONS}

CLASS_NAMES: List[str] = [
    "basophil",
    "eosinophil",
    "hairy_cell",
    "lymphocyte",
    "lymphocyte_large_granular",
    "lymphocyte_neoplastic",
    "metamyelocyte",
    "monocyte",
    "myeloblast",
    "myelocyte",
    "neutrophil_band",
    "neutrophil_segmented",
    "normoblast",
    "plasma_cell",
    "promyelocyte",
    "promyelocyte_atypical",
]
CLASS_NAME_TO_INDEX: Dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)


# Your state_dict files (already converted)
MODEL_STATE_DICT_PATHS: Dict[str, Path] = {
    "falcon_gold": PROJECT_ROOT / "falcon_gold_state_dict.pt",
    "falcon_silver": PROJECT_ROOT / "falcon_silver_state_dict.pt",
    "falcon_bronze": PROJECT_ROOT / "falcon_bronze_state_dict.pt",
    "iguana_gold": PROJECT_ROOT / "iguana_gold_state_dict.pt",
    "iguana_silver": PROJECT_ROOT / "iguana_silver_state_dict.pt",
    "iguana_bronze": PROJECT_ROOT / "iguana_bronze_state_dict.pt",
}

# Example files (optional; script can also scan TEST_ROOT)
EXAMPLE_IMAGE_PATHS: List[Path] = [
    PROJECT_ROOT / "blood_cancer_cells_split/te/hairy_cell/hairy_cell_0024_lab_c0_d0.jpg",
    PROJECT_ROOT / "blood_cancer_cells_split/te/eosinophil/eosinophil_0024_lab_c0_d0.jpg",
]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower().lstrip(".") in _ALLOWED_IMAGE_EXTENSION_SET


def crop_inset_resize(image: Image.Image, *, inset: int, size: int) -> Image.Image:
    rgb_image = image.convert("RGB") if image.mode != "RGB" else image

    width, height = rgb_image.size
    inset = max(0, int(inset))

    cropped_width = width - 2 * inset
    cropped_height = height - 2 * inset

    if cropped_width <= 1 or cropped_height <= 1:
        square_size = max(1, min(width, height))
        left = (width - square_size) // 2
        top = (height - square_size) // 2
        rgb_image = rgb_image.crop((left, top, left + square_size, top + square_size))
    else:
        rgb_image = rgb_image.crop((inset, inset, width - inset, height - inset))

    return rgb_image.resize((int(size), int(size)), resample=Image.Resampling.BILINEAR)


def load_preprocessed_image_tensor(*, image_path: Path, network_size: int, crop_inset: int) -> torch.Tensor:
    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")

    image = crop_inset_resize(image, inset=int(crop_inset), size=int(network_size))

    image_array = np.asarray(image).astype(np.float32) / 255.0  # [H,W,C]
    image_array = np.transpose(image_array, (2, 0, 1))          # [C,H,W]
    return torch.from_numpy(image_array)


# -----------------------------
# Model blocks
# -----------------------------

class GeneralizedMeanPooling(nn.Module):
    def __init__(self, initial_exponent: float = 3.0, epsilon: float = 1e-6):
        super().__init__()
        self.exponent = nn.Parameter(torch.ones(1) * float(initial_exponent))
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=self.epsilon)
        x = x.pow(self.exponent)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.pow(1.0 / self.exponent)
        return x


class SqueezeExcitationBlock(nn.Module):
    def __init__(self, number_of_channels: int, squeeze_ratio: int):
        super().__init__()
        reduced_channels = max(8, int(number_of_channels) // int(squeeze_ratio))

        self.global_average_pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(int(number_of_channels), int(reduced_channels), kernel_size=1)
        self.expand = nn.Conv2d(int(reduced_channels), int(number_of_channels), kernel_size=1)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gating = self.global_average_pool(x)
        gating = self.reduce(gating)
        gating = self.activation(gating)
        gating = self.expand(gating)
        gating = torch.sigmoid(gating)
        return x * gating


class DepthwiseSeparableBottleneckBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        group_normalization_groups: int,
        squeeze_ratio: int,
    ):
        super().__init__()

        if int(stride) != 1:
            raise ValueError(
                "DepthwiseSeparableBottleneckBlock requires stride == 1 "
                f"(got stride={stride}). Use DepthwiseSeparableProjectionBottleneckBlock for downsampling."
            )
        if int(input_channels) != int(output_channels):
            raise ValueError(
                "DepthwiseSeparableBottleneckBlock requires input_channels == output_channels "
                f"(got {input_channels} vs {output_channels}). Use DepthwiseSeparableProjectionBottleneckBlock for channel changes."
            )

        self.activation = nn.SiLU(inplace=False)

        self.depthwise_convolution = nn.Conv2d(
            int(input_channels),
            int(input_channels),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=int(input_channels),
            bias=False,
        )
        self.depthwise_normalization = nn.GroupNorm(
            num_groups=int(group_normalization_groups),
            num_channels=int(input_channels),
        )

        self.pointwise_convolution = nn.Conv2d(
            int(input_channels),
            int(output_channels),
            kernel_size=1,
            bias=False,
        )
        self.pointwise_normalization = nn.GroupNorm(
            num_groups=int(group_normalization_groups),
            num_channels=int(output_channels),
        )

        self.squeeze_excitation = SqueezeExcitationBlock(
            number_of_channels=int(output_channels),
            squeeze_ratio=int(squeeze_ratio),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        y = self.depthwise_convolution(x)
        y = self.depthwise_normalization(y)
        y = self.activation(y)

        y = self.pointwise_convolution(y)
        y = self.pointwise_normalization(y)
        y = self.activation(y)

        y = self.squeeze_excitation(y)
        return y + identity


class DepthwiseSeparableProjectionBottleneckBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        group_normalization_groups: int,
        squeeze_ratio: int,
    ):
        super().__init__()

        self.activation = nn.SiLU(inplace=False)

        self.depthwise_convolution = nn.Conv2d(
            int(input_channels),
            int(input_channels),
            kernel_size=3,
            stride=int(stride),
            padding=1,
            groups=int(input_channels),
            bias=False,
        )
        self.depthwise_normalization = nn.GroupNorm(
            num_groups=int(group_normalization_groups),
            num_channels=int(input_channels),
        )

        self.pointwise_convolution = nn.Conv2d(
            int(input_channels),
            int(output_channels),
            kernel_size=1,
            bias=False,
        )
        self.pointwise_normalization = nn.GroupNorm(
            num_groups=int(group_normalization_groups),
            num_channels=int(output_channels),
        )

        self.squeeze_excitation = SqueezeExcitationBlock(
            number_of_channels=int(output_channels),
            squeeze_ratio=int(squeeze_ratio),
        )

        self.projection_skip = nn.Sequential(
            nn.Conv2d(
                int(input_channels),
                int(output_channels),
                kernel_size=1,
                stride=int(stride),
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=int(group_normalization_groups),
                num_channels=int(output_channels),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.projection_skip(x)

        y = self.depthwise_convolution(x)
        y = self.depthwise_normalization(y)
        y = self.activation(y)

        y = self.pointwise_convolution(y)
        y = self.pointwise_normalization(y)
        y = self.activation(y)

        y = self.squeeze_excitation(y)
        return y + identity


# Iguana fusion parts

class DualPoolingProjection(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dimension: int,
        initial_pooling_exponent: float,
    ):
        super().__init__()

        self.generalized_mean_pooling = GeneralizedMeanPooling(
            initial_exponent=float(initial_pooling_exponent)
        )
        self.average_pooling = nn.AdaptiveAvgPool2d(1)

        self.projection = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(int(input_channels) * 2, int(embedding_dimension)),
            nn.SiLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        generalized_mean = self.generalized_mean_pooling(x)
        average = self.average_pooling(x)
        stacked = torch.cat([generalized_mean, average], dim=1)  # [N, 2C, 1, 1]
        return self.projection(stacked)


class AuxiliaryKnowledgeBranch(nn.Module):
    def __init__(
        self,
        input_channels: int,
        embedding_dimension: int,
        initial_pooling_exponent: float,
        dropout_probability: float,
    ):
        super().__init__()

        self.pooling = GeneralizedMeanPooling(initial_exponent=float(initial_pooling_exponent))

        self.projection = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(int(input_channels), int(embedding_dimension)),
            nn.SiLU(inplace=False),
            nn.Dropout(p=float(dropout_probability)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pooling(x)
        return self.projection(x)


class GatedFeatureFusionClassifier(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        number_of_classes: int,
        dropout_probability: float,
    ):
        super().__init__()

        self.gating_network = nn.Sequential(
            nn.Linear(int(embedding_dimension) * 2, int(embedding_dimension)),
            nn.SiLU(inplace=False),
            nn.Linear(int(embedding_dimension), int(embedding_dimension)),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(embedding_dimension), int(embedding_dimension // 2)),
            nn.SiLU(inplace=False),
            nn.Dropout(p=float(dropout_probability)),
            nn.Linear(int(embedding_dimension // 2), int(number_of_classes)),
        )

    def forward(self, main_embedding: torch.Tensor, auxiliary_embedding: torch.Tensor) -> torch.Tensor:
        gating_values = self.gating_network(torch.cat([main_embedding, auxiliary_embedding], dim=1))
        fused_embedding = (gating_values * main_embedding) + ((1.0 - gating_values) * auxiliary_embedding)
        return self.classifier(fused_embedding)


# Models (do not rename)

class Iguana64(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
        dropout_probability: float = 0.25,
        group_normalization_groups: int = 8,
        initial_pooling_exponent: float = 3.0,
        squeeze_ratio: int = 32,
        stem_channels: int = 64,
        stage_two_channels: int = 128,
        stage_three_channels: int = 256,
        stage_four_channels: int = 512,
        embedding_dimension: int = 128,
    ):
        super().__init__()

        activation = nn.SiLU(inplace=False)

        self.stem = nn.Sequential(
            nn.Conv2d(3, int(stem_channels), kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=int(group_normalization_groups), num_channels=int(stem_channels)),
            activation,
        )

        self.stage_two = nn.Sequential(
            DepthwiseSeparableProjectionBottleneckBlock(
                input_channels=int(stem_channels),
                output_channels=int(stage_two_channels),
                stride=2,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
            DepthwiseSeparableBottleneckBlock(
                input_channels=int(stage_two_channels),
                output_channels=int(stage_two_channels),
                stride=1,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
        )

        self.stage_three = nn.Sequential(
            DepthwiseSeparableProjectionBottleneckBlock(
                input_channels=int(stage_two_channels),
                output_channels=int(stage_three_channels),
                stride=2,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
            DepthwiseSeparableBottleneckBlock(
                input_channels=int(stage_three_channels),
                output_channels=int(stage_three_channels),
                stride=1,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
        )

        self.stage_four = nn.Sequential(
            DepthwiseSeparableProjectionBottleneckBlock(
                input_channels=int(stage_three_channels),
                output_channels=int(stage_four_channels),
                stride=2,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
            DepthwiseSeparableBottleneckBlock(
                input_channels=int(stage_four_channels),
                output_channels=int(stage_four_channels),
                stride=1,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
        )

        self.main_dual_pooling_projection = DualPoolingProjection(
            input_channels=int(stage_four_channels),
            embedding_dimension=int(embedding_dimension),
            initial_pooling_exponent=float(initial_pooling_exponent),
        )

        self.auxiliary_branch = AuxiliaryKnowledgeBranch(
            input_channels=int(stage_three_channels),
            embedding_dimension=int(embedding_dimension),
            initial_pooling_exponent=float(initial_pooling_exponent),
            dropout_probability=float(dropout_probability),
        )

        self.fusion_classifier = GatedFeatureFusionClassifier(
            embedding_dimension=int(embedding_dimension),
            number_of_classes=int(number_of_classes),
            dropout_probability=float(dropout_probability),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        stage_two_features = self.stage_two(x)
        stage_three_features = self.stage_three(stage_two_features)
        stage_four_features = self.stage_four(stage_three_features)
        main_embedding = self.main_dual_pooling_projection(stage_four_features)
        auxiliary_embedding = self.auxiliary_branch(stage_three_features)
        return self.fusion_classifier(main_embedding, auxiliary_embedding)


class Falcon64(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
        dropout_probability: float = 0.25,
        group_normalization_groups: int = 8,
        initial_pooling_exponent: float = 3.0,
        squeeze_ratio: int = 32,
        stem_channels: int = 64,
        stage_two_channels: int = 128,
        stage_three_channels: int = 256,
        stage_four_channels: int = 512,
        head_hidden_dimension: int = 64,
    ):
        super().__init__()

        activation = nn.SiLU(inplace=False)

        self.stem = nn.Sequential(
            nn.Conv2d(3, int(stem_channels), kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=int(group_normalization_groups), num_channels=int(stem_channels)),
            activation,
        )

        self.stage_two = nn.Sequential(
            DepthwiseSeparableProjectionBottleneckBlock(
                input_channels=int(stem_channels),
                output_channels=int(stage_two_channels),
                stride=2,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
            DepthwiseSeparableBottleneckBlock(
                input_channels=int(stage_two_channels),
                output_channels=int(stage_two_channels),
                stride=1,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
        )

        self.stage_three = nn.Sequential(
            DepthwiseSeparableProjectionBottleneckBlock(
                input_channels=int(stage_two_channels),
                output_channels=int(stage_three_channels),
                stride=2,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
            DepthwiseSeparableBottleneckBlock(
                input_channels=int(stage_three_channels),
                output_channels=int(stage_three_channels),
                stride=1,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
        )

        self.stage_four = nn.Sequential(
            DepthwiseSeparableProjectionBottleneckBlock(
                input_channels=int(stage_three_channels),
                output_channels=int(stage_four_channels),
                stride=2,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
            DepthwiseSeparableBottleneckBlock(
                input_channels=int(stage_four_channels),
                output_channels=int(stage_four_channels),
                stride=1,
                group_normalization_groups=int(group_normalization_groups),
                squeeze_ratio=int(squeeze_ratio),
            ),
        )

        self.pooling = GeneralizedMeanPooling(initial_exponent=float(initial_pooling_exponent))

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(int(stage_four_channels), int(head_hidden_dimension)),
            nn.SiLU(inplace=False),
            nn.Dropout(p=float(dropout_probability)),
            nn.Linear(int(head_hidden_dimension), int(number_of_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage_two(x)
        x = self.stage_three(x)
        x = self.stage_four(x)
        x = self.pooling(x)
        logits = self.classifier(x)
        return logits


# -----------------------------
# Ensemble configuration
# -----------------------------

@dataclass(frozen=True)
class ModelConfiguration:
    architecture_name: str
    member_name: str
    group_normalization_groups: int
    dropout_probability: float
    embedding_dimension: int  # falcon head_hidden_dimension OR iguana embedding_dimension
    state_dict_path: Path


FALCON_GOLD = ModelConfiguration("falcon64", "falcon_gold",   32, 0.2500,  96, MODEL_STATE_DICT_PATHS["falcon_gold"])
FALCON_SILVER = ModelConfiguration("falcon64", "falcon_silver", 16, 0.1500,  96, MODEL_STATE_DICT_PATHS["falcon_silver"])
FALCON_BRONZE = ModelConfiguration("falcon64", "falcon_bronze", 32, 0.3000,  96, MODEL_STATE_DICT_PATHS["falcon_bronze"])

IGUANA_GOLD = ModelConfiguration("iguana64", "iguana_gold",   16, 0.2000, 256, MODEL_STATE_DICT_PATHS["iguana_gold"])
IGUANA_SILVER = ModelConfiguration("iguana64", "iguana_silver",  4, 0.2000, 128, MODEL_STATE_DICT_PATHS["iguana_silver"])
IGUANA_BRONZE = ModelConfiguration("iguana64", "iguana_bronze", 16, 0.2000, 256, MODEL_STATE_DICT_PATHS["iguana_bronze"])

ENSEMBLE_MEMBERS: List[ModelConfiguration] = [
    FALCON_GOLD,
    FALCON_SILVER,
    FALCON_BRONZE,
    IGUANA_GOLD,
    IGUANA_SILVER,
    IGUANA_BRONZE,
]


def instantiate_model_from_configuration(member: ModelConfiguration) -> nn.Module:
    if member.architecture_name == "falcon64":
        return Falcon64(
            number_of_classes=int(NUM_CLASSES),
            dropout_probability=float(member.dropout_probability),
            group_normalization_groups=int(member.group_normalization_groups),
            initial_pooling_exponent=3.0,
            squeeze_ratio=int(SQUEEZE_RATIO),
            stem_channels=64,
            stage_two_channels=128,
            stage_three_channels=256,
            stage_four_channels=512,
            head_hidden_dimension=int(member.embedding_dimension),
        )
    if member.architecture_name == "iguana64":
        return Iguana64(
            number_of_classes=int(NUM_CLASSES),
            dropout_probability=float(member.dropout_probability),
            group_normalization_groups=int(member.group_normalization_groups),
            initial_pooling_exponent=3.0,
            squeeze_ratio=int(SQUEEZE_RATIO),
            stem_channels=64,
            stage_two_channels=128,
            stage_three_channels=256,
            stage_four_channels=512,
            embedding_dimension=int(member.embedding_dimension),
        )
    raise ValueError(f"Unknown architecture_name: {member.architecture_name}")


def load_state_dict_strict(*, model: nn.Module, state_dict_path: Path, device: torch.device) -> nn.Module:
    state_dict = torch.load(str(state_dict_path), map_location=device)

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Expected a raw state_dict dict at: {state_dict_path}")

    if any(str(key).startswith("module.") for key in state_dict.keys()):
        state_dict = {str(key).replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_ensemble_models(*, device: torch.device) -> Dict[str, nn.Module]:
    member_models: Dict[str, nn.Module] = {}

    print("🧩 Loading ensemble members (6):")
    for member in ENSEMBLE_MEMBERS:
        if not member.state_dict_path.exists():
            raise FileNotFoundError(f"Missing state_dict: {member.state_dict_path}")

        model = instantiate_model_from_configuration(member)
        model = load_state_dict_strict(model=model, state_dict_path=member.state_dict_path, device=device)

        member_models[member.member_name] = model
        size_mb = member.state_dict_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ {member.member_name:13s} | {member.architecture_name:7s} | {size_mb:5.2f} MB | {member.state_dict_path.name}")

    return member_models


@torch.inference_mode()
def run_ensemble_on_single_image(
    *,
    device: torch.device,
    member_models: Dict[str, nn.Module],
    image_tensor: torch.Tensor,  # [C,H,W] in [0,1]
) -> Dict[str, object]:
    member_items = sorted(member_models.items(), key=lambda item: str(item[0]))
    batch = image_tensor.unsqueeze(0).to(device)  # [1,C,H,W]

    per_member_top1_index: Dict[str, int] = {}
    per_member_top1_probability: Dict[str, float] = {}

    probability_sum: Optional[torch.Tensor] = None

    for member_name, model in member_items:
        logits = model(batch)               # [1,C]
        probabilities = F.softmax(logits, dim=1)  # [1,C]

        probability_sum = probabilities if probability_sum is None else (probability_sum + probabilities)

        probs_cpu = probabilities[0].detach().cpu().numpy().astype(np.float64)  # [C]
        top1_index = int(np.argmax(probs_cpu))
        top1_prob = float(probs_cpu[top1_index])

        per_member_top1_index[str(member_name)] = int(top1_index)
        per_member_top1_probability[str(member_name)] = float(top1_prob)

    mean_probabilities = (probability_sum / float(len(member_items)))[0]  # [C]
    ensemble_probs = mean_probabilities.detach().cpu().numpy().astype(np.float64)

    ensemble_top1_index = int(np.argmax(ensemble_probs))
    ensemble_top1_probability = float(ensemble_probs[ensemble_top1_index])

    return {
        "per_member_top1_index": per_member_top1_index,
        "per_member_top1_probability": per_member_top1_probability,
        "ensemble_top1_index": ensemble_top1_index,
        "ensemble_top1_probability": ensemble_top1_probability,
        "member_names_sorted": [name for name, _ in member_items],
    }


def infer_true_label_from_path(image_path: Path) -> Optional[str]:
    # Your folder structure is .../te/<class_name>/<file>.jpg
    # If it matches known classes, return it; otherwise None.
    if len(image_path.parts) < 2:
        return None
    parent = image_path.parent.name
    if parent in CLASS_NAME_TO_INDEX:
        return parent
    return None


def iter_test_images(*, test_root: Path) -> List[Path]:
    if not test_root.exists():
        raise FileNotFoundError(f"TEST_ROOT not found: {test_root}")
    return sorted([p for p in test_root.rglob("*") if is_image_file(p)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--test_root", type=str, default=str(TEST_ROOT))
    parser.add_argument("--image", type=str, action="append", default=[])
    parser.add_argument("--limit", type=int, default=0, help="If >0, only run the first N images.")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    test_root = Path(args.test_root)

    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)

    print("🧠 torch:", torch.__version__)
    print("🖥️ device:", device)
    print("📦 project_root:", project_root)
    print("🧪 test_root:", test_root)

    member_models = load_ensemble_models(device=device)

    explicit_images = [Path(p) for p in args.image]
    if explicit_images:
        image_paths = explicit_images
        print(f"\n🖼️ Using explicit --image paths ({len(image_paths)}):")
        for p in image_paths:
            print("  -", p)
    else:
        # If no explicit list, default to scanning test_root.
        image_paths = iter_test_images(test_root=test_root)
        if not image_paths:
            # fallback to the two example paths if test_root is missing/empty
            image_paths = [p for p in EXAMPLE_IMAGE_PATHS if p.exists()]

        print(f"\n🖼️ Scanning test set images: {len(image_paths)} found.")

    limit = int(args.limit)
    if limit > 0:
        image_paths = image_paths[:limit]
        print(f"🔎 limit applied: {len(image_paths)} images")

    print("\n🏁 Running inference...\n")

    correct_count = 0
    total_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        if not image_path.exists():
            print(f"[{index:05d}] ❌ missing: {image_path}")
            continue

        image_tensor = load_preprocessed_image_tensor(
            image_path=image_path,
            network_size=int(NETWORK_SIZE),
            crop_inset=int(NETWORK_POST_AUGMENTATION_INSET),
        )

        prediction = run_ensemble_on_single_image(
            device=device,
            member_models=member_models,
            image_tensor=image_tensor,
        )

        ensemble_top1_index = int(prediction["ensemble_top1_index"])
        ensemble_top1_probability = float(prediction["ensemble_top1_probability"])
        ensemble_name = str(CLASS_NAMES[ensemble_top1_index])

        true_label = infer_true_label_from_path(image_path)
        is_correct = (true_label == ensemble_name) if true_label is not None else None

        per_member_top1_index: Dict[str, int] = dict(prediction["per_member_top1_index"])
        per_member_top1_probability: Dict[str, float] = dict(prediction["per_member_top1_probability"])
        member_names_sorted: List[str] = list(prediction["member_names_sorted"])

        # Header per image
        total_count += 1
        correctness_icon = ""
        if is_correct is True:
            correctness_icon = " ✅"
            correct_count += 1
        elif is_correct is False:
            correctness_icon = " ❌"

        true_text = f"true={true_label}" if true_label is not None else "true=?"
        print(f"[{index:05d}] {image_path.name} | {true_text} | ensemble={ensemble_name} p={ensemble_top1_probability:.3f}{correctness_icon}")

        # Per-member lines
        for member_name in member_names_sorted:
            pred_index = int(per_member_top1_index[member_name])
            pred_prob = float(per_member_top1_probability[member_name])
            pred_name = str(CLASS_NAMES[pred_index])

            member_icon = ""
            if true_label is not None:
                member_icon = "✅" if pred_name == true_label else "❌"

            print(f"         {member_icon:2s} {member_name:13s} -> {pred_name:28s} p={pred_prob:.3f}")

        print("")

    if total_count > 0:
        if correct_count > 0:
            accuracy = float(correct_count) / float(total_count)
            print(f"📈 Ensemble top1 accuracy (folder-label heuristic): {accuracy:.4f} ({correct_count}/{total_count})")
        else:
            print("📌 Done. (No folder-based labels matched, so accuracy summary skipped.)")


if __name__ == "__main__":
    main()