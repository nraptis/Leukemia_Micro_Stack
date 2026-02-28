# ml/labels.py

from __future__ import annotations
from typing import Dict, List

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
NUMBER_OF_CLASSES: int = len(CLASS_NAMES)

def class_index(name: str) -> int:
    normalized_name = str(name).strip().lower()
    if normalized_name not in CLASS_NAME_TO_INDEX:
        raise KeyError(f"Unknown class name: {name!r}")
    return int(CLASS_NAME_TO_INDEX[normalized_name])


def index_class(index: int) -> str:
    index = int(index)
    if index < 0 or index >= len(CLASS_NAMES):
        raise IndexError(f"Class index out of range: {index}")
    return str(CLASS_NAMES[index])