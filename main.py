# main.py

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ml.classify import classify
from ml.pretty_print import to_pretty_print

ROOT_DIRECTORY = Path(__file__).resolve().parent

IMAGE_PATHS = [
    ROOT_DIRECTORY / "images" / "myeloblast_0014_lab_c0_d0.jpg",
    ROOT_DIRECTORY / "images" / "myeloblast_0024_lab_c0_d0.jpg",
    ROOT_DIRECTORY / "images" / "myeloblast_0022_lab_c0_d0.jpg",
]

def _format_probability(probability: float) -> str:
    return f"{probability * 100.0:6.2f}%"

def main() -> None:
    
    for image_path in IMAGE_PATHS:
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

    for image_path in IMAGE_PATHS:
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")

        result = classify(image=image, normalize=True)

        pretty = to_pretty_print(
            result,
            top_k_per_model=3,
            top_k_ensemble=10,
            allow_ties_per_model=True,
            allow_ties_ensemble=True,
        )

        print(f"\n\n🖼️🖼️🖼️🖼️🖼️🖼️ {image_path.name} 🖼️🖼️🖼️🖼️🖼️🖼️")

        for model_name in pretty.model_list:
            entry = pretty.model_result_table[model_name]
            parts = [f"{cls}={_format_probability(entry.class_table[cls])}" for cls in entry.class_list]
            print(f"  {model_name:13s} | " + ", ".join(parts))

        print("\nENSEMBLE:\n")
        for leet_ITAM in pretty.ensemble_list:
            print(f"\t{leet_ITAM.class_name}={_format_probability(leet_ITAM.probability)}")


if __name__ == "__main__":
    main()