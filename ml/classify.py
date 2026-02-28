# ml/classify.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.preprocess import preprocess
from ml.labels import NUMBER_OF_CLASSES, class_index
from ml.models import (
    load_falcon_gold,
    load_falcon_silver,
    load_falcon_bronze,
    load_iguana_gold,
    load_iguana_silver,
    load_iguana_bronze,
)

ENSEMBLE_MEMBER_LOADERS = {
    "falcon_gold": load_falcon_gold,
    "falcon_silver": load_falcon_silver,
    "falcon_bronze": load_falcon_bronze,
    "iguana_gold": load_iguana_gold,
    "iguana_silver": load_iguana_silver,
    "iguana_bronze": load_iguana_bronze,
}


# -----------------------------
# Result Types (Raw Inference)
# -----------------------------

@dataclass(frozen=True)
class ClassifyResultVector:
    probabilities_by_index: np.ndarray  # shape [16], float64

    def get(self, name: str) -> float:
        index = class_index(name)
        return float(self.probabilities_by_index[index])

    def sum(self) -> float:
        return float(self.probabilities_by_index.sum())


@dataclass(frozen=True)
class ClassifyResult:
    falcon_gold: ClassifyResultVector
    falcon_silver: ClassifyResultVector
    falcon_bronze: ClassifyResultVector
    iguana_gold: ClassifyResultVector
    iguana_silver: ClassifyResultVector
    iguana_bronze: ClassifyResultVector
    ensemble_mean: ClassifyResultVector

    def member(self, name: str) -> ClassifyResultVector:
        normalized_name = str(name).strip().lower()
        mapping = {
            "falcon_gold": self.falcon_gold,
            "falcon_silver": self.falcon_silver,
            "falcon_bronze": self.falcon_bronze,
            "iguana_gold": self.iguana_gold,
            "iguana_silver": self.iguana_silver,
            "iguana_bronze": self.iguana_bronze,
            "ensemble_mean": self.ensemble_mean,
        }
        if normalized_name not in mapping:
            raise KeyError(f"Unknown member: {name!r}")
        return mapping[normalized_name]


# -----------------------------
# Inference Helpers
# -----------------------------

def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")

    array = np.asarray(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)

_ensemble_models_cache: Optional[Dict[str, nn.Module]] = None

def _get_device() -> torch.device:
    return torch.device("cpu")

def _load_ensemble_models_if_needed(device: torch.device) -> Dict[str, nn.Module]:
    global _ensemble_models_cache

    if _ensemble_models_cache is not None:
        return _ensemble_models_cache

    models: Dict[str, nn.Module] = {}

    for name, loader in ENSEMBLE_MEMBER_LOADERS.items():
        model = loader(
            number_of_classes=int(NUMBER_OF_CLASSES),
            device=device,
        )
        models[name] = model

    _ensemble_models_cache = models
    return models

@torch.inference_mode()
def _run_member(
    device: torch.device,
    model: nn.Module,
    image_tensor: torch.Tensor,
) -> np.ndarray:
    batch = image_tensor.unsqueeze(0).to(device)
    logits = model(batch)
    probabilities = F.softmax(logits, dim=1)[0]
    return probabilities.detach().cpu().numpy().astype(np.float64)

# -----------------------------
# Public API
# -----------------------------

def classify(image: Image.Image, normalize: bool) -> ClassifyResult:
    device = _get_device()
    models = _load_ensemble_models_if_needed(device=device)

    processed_image = preprocess(image=image, normalize=bool(normalize))
    image_tensor = _image_to_tensor(processed_image)

    member_vectors: Dict[str, ClassifyResultVector] = {}
    probability_sum = np.zeros((int(NUMBER_OF_CLASSES),), dtype=np.float64)

    for name, model in models.items():
        probs = _run_member(device=device, model=model, image_tensor=image_tensor)
        probability_sum += probs
        member_vectors[name] = ClassifyResultVector(probs)

    ensemble_mean = probability_sum / float(len(models))
    ensemble_mean = ensemble_mean / ensemble_mean.sum()  # exact normalization

    return ClassifyResult(
        falcon_gold=member_vectors["falcon_gold"],
        falcon_silver=member_vectors["falcon_silver"],
        falcon_bronze=member_vectors["falcon_bronze"],
        iguana_gold=member_vectors["iguana_gold"],
        iguana_silver=member_vectors["iguana_silver"],
        iguana_bronze=member_vectors["iguana_bronze"],
        ensemble_mean=ClassifyResultVector(ensemble_mean),
    )