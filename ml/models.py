# ml/models.py

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Only expose the 6 member loaders from this module.
__all__ = [
    "load_falcon_gold",
    "load_falcon_silver",
    "load_falcon_bronze",
    "load_iguana_gold",
    "load_iguana_silver",
    "load_iguana_bronze",
]

_THIS_DIRECTORY = Path(__file__).resolve().parent
STATE_DICT_DIRECTORY = _THIS_DIRECTORY / "state_dicts"

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
            raise ValueError("BottleneckBlock requires stride == 1.")
        if int(input_channels) != int(output_channels):
            raise ValueError("BottleneckBlock requires input_channels == output_channels.")

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
    def __init__(self, input_channels: int, embedding_dimension: int, initial_pooling_exponent: float):
        super().__init__()
        self.generalized_mean_pooling = GeneralizedMeanPooling(initial_exponent=float(initial_pooling_exponent))
        self.average_pooling = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(int(input_channels) * 2, int(embedding_dimension)),
            nn.SiLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        generalized_mean = self.generalized_mean_pooling(x)
        average = self.average_pooling(x)
        stacked = torch.cat([generalized_mean, average], dim=1)  # [N,2C,1,1]
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
    def __init__(self, embedding_dimension: int, number_of_classes: int, dropout_probability: float):
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

# -----------------------------
# Architectures
# -----------------------------

class Iguana64(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
        dropout_probability: float,
        group_normalization_groups: int,
        embedding_dimension: int,
        initial_pooling_exponent: float = 3.0,
        squeeze_ratio: int = 32,
        stem_channels: int = 64,
        stage_two_channels: int = 128,
        stage_three_channels: int = 256,
        stage_four_channels: int = 512,
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
        dropout_probability: float,
        group_normalization_groups: int,
        head_hidden_dimension: int,
        initial_pooling_exponent: float = 3.0,
        squeeze_ratio: int = 32,
        stem_channels: int = 64,
        stage_two_channels: int = 128,
        stage_three_channels: int = 256,
        stage_four_channels: int = 512,
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
        return self.classifier(x)
    
@dataclass(frozen=True)
class _MemberConfiguration:
    architecture_name: str
    member_name: str
    group_normalization_groups: int
    dropout_probability: float
    embedding_dimension: int
    state_dict_path: Path

def _instantiate_model(*, member: _MemberConfiguration, number_of_classes: int) -> nn.Module:
    if member.architecture_name == "falcon64":
        return Falcon64(
            number_of_classes=int(number_of_classes),
            dropout_probability=float(member.dropout_probability),
            group_normalization_groups=int(member.group_normalization_groups),
            head_hidden_dimension=int(member.embedding_dimension),
        )

    if member.architecture_name == "iguana64":
        return Iguana64(
            number_of_classes=int(number_of_classes),
            dropout_probability=float(member.dropout_probability),
            group_normalization_groups=int(member.group_normalization_groups),
            embedding_dimension=int(member.embedding_dimension),
        )

    raise ValueError(f"Unknown architecture_name: {member.architecture_name}")

def _load_state_dict_strict(*, model: nn.Module, state_dict_path: Path, device: torch.device) -> nn.Module:
    state_dict = torch.load(str(state_dict_path), map_location=device)

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Expected a raw state_dict dict at: {state_dict_path}")
    
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def _load_member(*, member: _MemberConfiguration, number_of_classes: int, device: torch.device) -> nn.Module:
    if not member.state_dict_path.exists():
        raise FileNotFoundError(f"Missing state_dict: {member.state_dict_path}")

    model = _instantiate_model(member=member, number_of_classes=int(number_of_classes))
    model = _load_state_dict_strict(model=model, state_dict_path=member.state_dict_path, device=device)
    return model

def load_falcon_gold(number_of_classes: int, device: torch.device) -> nn.Module:
    member = _MemberConfiguration(
        architecture_name="falcon64",
        member_name="falcon_gold",
        group_normalization_groups=32,
        dropout_probability=0.2500,
        embedding_dimension=96,
        state_dict_path=STATE_DICT_DIRECTORY / "falcon_gold_state_dict.pt",
    )
    return _load_member(member=member, number_of_classes=int(number_of_classes), device=device)

def load_falcon_silver(number_of_classes: int, device: torch.device) -> nn.Module:
    member = _MemberConfiguration(
        architecture_name="falcon64",
        member_name="falcon_silver",
        group_normalization_groups=16,
        dropout_probability=0.1500,
        embedding_dimension=96,
        state_dict_path=STATE_DICT_DIRECTORY / "falcon_silver_state_dict.pt",
    )
    return _load_member(member=member, number_of_classes=int(number_of_classes), device=device)

def load_falcon_bronze(number_of_classes: int, device: torch.device) -> nn.Module:
    member = _MemberConfiguration(
        architecture_name="falcon64",
        member_name="falcon_bronze",
        group_normalization_groups=32,
        dropout_probability=0.3000,
        embedding_dimension=96,
        state_dict_path=STATE_DICT_DIRECTORY / "falcon_bronze_state_dict.pt",
    )
    return _load_member(member=member, number_of_classes=int(number_of_classes), device=device)

def load_iguana_gold(number_of_classes: int, device: torch.device) -> nn.Module:
    member = _MemberConfiguration(
        architecture_name="iguana64",
        member_name="iguana_gold",
        group_normalization_groups=16,
        dropout_probability=0.2000,
        embedding_dimension=256,
        state_dict_path=STATE_DICT_DIRECTORY / "iguana_gold_state_dict.pt",
    )
    return _load_member(member=member, number_of_classes=int(number_of_classes), device=device)


def load_iguana_silver(number_of_classes: int, device: torch.device) -> nn.Module:
    member = _MemberConfiguration(
        architecture_name="iguana64",
        member_name="iguana_silver",
        group_normalization_groups=4,
        dropout_probability=0.2000,
        embedding_dimension=128,
        state_dict_path=STATE_DICT_DIRECTORY / "iguana_silver_state_dict.pt",
    )
    return _load_member(member=member, number_of_classes=int(number_of_classes), device=device)


def load_iguana_bronze(number_of_classes: int, device: torch.device) -> nn.Module:
    member = _MemberConfiguration(
        architecture_name="iguana64",
        member_name="iguana_bronze",
        group_normalization_groups=16,
        dropout_probability=0.2000,
        embedding_dimension=256,
        state_dict_path=STATE_DICT_DIRECTORY / "iguana_bronze_state_dict.pt",
    )
    return _load_member(member=member, number_of_classes=int(number_of_classes), device=device)