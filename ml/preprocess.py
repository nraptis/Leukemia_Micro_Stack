# preprocess.py

import numpy as np
from PIL import Image
import cv2

LAB_NORMALIZATION_TARGET_MEAN = 145.0
LAB_NORMALIZATION_TARGET_STD = 40.0

__all__ = ["preprocess"]

def _center_square_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    crop_size = int(min(size, width, height))
    left = int(round((width - crop_size) / 2.0))
    top = int(round((height - crop_size) / 2.0))
    right = left + crop_size
    bottom = top + crop_size
    return image.crop((left, top, right, bottom))

def _normalize(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    rgb_uint8 = np.asarray(image)
    lab_float32 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)

    l_channel_float32, a_channel_float32, b_channel_float32 = cv2.split(lab_float32)

    epsilon = 1e-6
    luminance_mean = float(l_channel_float32.mean())
    luminance_std = float(l_channel_float32.std()) + epsilon

    l_channel_float32 = (
        (l_channel_float32 - luminance_mean)
        / luminance_std
        * float(LAB_NORMALIZATION_TARGET_STD)
        + float(LAB_NORMALIZATION_TARGET_MEAN)
    )
    l_channel_uint8 = np.clip(l_channel_float32, 0.0, 255.0).astype(np.uint8)
    a_channel_uint8 = np.clip(a_channel_float32, 0.0, 255.0).astype(np.uint8)
    b_channel_uint8 = np.clip(b_channel_float32, 0.0, 255.0).astype(np.uint8)
    rgb2_uint8 = cv2.cvtColor(
        cv2.merge([l_channel_uint8, a_channel_uint8, b_channel_uint8]),
        cv2.COLOR_LAB2RGB,
    )
    return Image.fromarray(rgb2_uint8)

def preprocess(image: Image.Image, normalize: bool) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")

    if normalize:
        image = _normalize(image)

    width, height = image.size

    if width != height:
        crop_size = int(min(width, height))
        image = _center_square_crop(image=image, size=crop_size)
        width, height = image.size  # refresh dimensions

    if not (width == 224 and height == 224):
        image = image.resize((224, 224), resample=Image.BILINEAR)

    return image