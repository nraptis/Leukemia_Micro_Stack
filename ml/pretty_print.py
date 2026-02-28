from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ml.labels import CLASS_NAMES
from ml.classify import ClassifyResult


PRETTY_MODEL_ORDER: List[str] = [
    "iguana_gold",
    "iguana_silver",
    "iguana_bronze",
    "falcon_gold",
    "falcon_silver",
    "falcon_bronze",
]

# -----------------------------
# Pretty Dataclasses
# -----------------------------

@dataclass(frozen=True)
class PrettyPrintSingleModelResult:
    class_list: List[str]
    class_table: Dict[str, float]


@dataclass(frozen=True)
class PrettyPrintSingleItemResult:
    class_name: str
    probability: float


@dataclass(frozen=True)
class PrettyPrintResult:
    model_result_table: Dict[str, PrettyPrintSingleModelResult]
    model_list: List[str]
    ensemble_list: List[PrettyPrintSingleItemResult]


# -----------------------------
# Ranking Helpers
# -----------------------------

def _sorted_items(probabilities: np.ndarray) -> List[Tuple[str, float]]:
    items = [
        (CLASS_NAMES[i], float(probabilities[i]))
        for i in range(len(CLASS_NAMES))
    ]
    items.sort(key=lambda pair: (-pair[1], pair[0]))
    return items


def _top_k_with_ties(
    items: List[Tuple[str, float]],
    k: int,
    allow_ties: bool,
    epsilon: float = 1e-12,
) -> List[Tuple[str, float]]:

    if len(items) <= k:
        return items

    if not allow_ties:
        return items[:k]

    cutoff = items[k - 1][1]
    selected = []

    for name, prob in items:
        if len(selected) < k:
            selected.append((name, prob))
        elif abs(prob - cutoff) <= epsilon:
            selected.append((name, prob))
        else:
            break

    return selected


# -----------------------------
# Public Formatting Function
# -----------------------------

def to_pretty_print(
    result: ClassifyResult,
    *,
    top_k_per_model: int = 3,
    top_k_ensemble: int = 10,
    allow_ties_per_model: bool = True,
    allow_ties_ensemble: bool = True,
) -> PrettyPrintResult:

    model_table: Dict[str, PrettyPrintSingleModelResult] = {}

    for model_name in PRETTY_MODEL_ORDER:
        vector = result.member(model_name)
        sorted_items = _sorted_items(vector.probabilities_by_index)
        selected = _top_k_with_ties(
            sorted_items,
            top_k_per_model,
            allow_ties_per_model,
        )

        model_table[model_name] = PrettyPrintSingleModelResult(
            class_list=[name for name, _ in selected],
            class_table={name: prob for name, prob in selected},
        )

    ensemble_sorted = _sorted_items(result.ensemble_mean.probabilities_by_index)
    ensemble_selected = _top_k_with_ties(
        ensemble_sorted,
        top_k_ensemble,
        allow_ties_ensemble,
    )

    ensemble_list = [
        PrettyPrintSingleItemResult(name, prob)
        for name, prob in ensemble_selected
    ]

    return PrettyPrintResult(
        model_result_table=model_table,
        model_list=PRETTY_MODEL_ORDER,
        ensemble_list=ensemble_list,
    )