"""
Image preprocessing utilities for image-based steganography
(autoencoder + LSB).
"""

import numpy as np
import cv2
from PIL import Image


# ------------------ BASIC LOAD / SAVE ------------------

def load_image(path: str) -> np.ndarray:
    """
    Load image in RGB format as numpy array (0–255 uint8).
    """
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save numpy image to disk (expects 0–255 uint8).
    """
    img = Image.fromarray(image.astype(np.uint8))
    img.save(path, quality=100)


# ------------------ RESIZING HELPERS ------------------

def resize_keep_aspect(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    """
    Resize while keeping aspect ratio — for COVER image capacity control.
    """
    h, w = image.shape[:2]

    if max(h, w) <= max_side:
        return image

    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_for_autoencoder(image: np.ndarray, target: int = 64) -> np.ndarray:
    """
    Resize secret image to the autoencoder input size.
    Keeps autoencoder training consistency.
    """
    return cv2.resize(image, (target, target), interpolation=cv2.INTER_AREA)


# ------------------ VALIDATION ------------------

def validate_image(image: np.ndarray) -> bool:
    """
    Validate 3-channel image for steganography.
    """
    return (
        image is not None
        and len(image.shape) == 3
        and image.shape[2] == 3
    )


# ------------------ NORMALIZATION FOR AUTOENCODER ------------------

def normalize_for_model(image: np.ndarray) -> np.ndarray:
    """
    Convert 0–255 uint8 → 0–1 float32
    """
    return image.astype("float32") / 255.0


def denormalize_from_model(image: np.ndarray) -> np.ndarray:
    """
    Convert model output 0–1 float → 0–255 uint8
    """
    image = np.clip(image * 255.0, 0, 255)
    return image.astype(np.uint8)


# ------------------ DIFF VISUALIZATION ------------------

def compare_images(original: np.ndarray, stego: np.ndarray) -> np.ndarray:
    """
    Compute amplified difference map for visualization.
    """
    diff = cv2.absdiff(original, stego)
    diff = np.clip(diff * 30, 0, 255).astype(np.uint8)
    return diff


# ------------------ CONVERSION HELPERS ------------------

def to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert BGR (OpenCV) → RGB (model/plotting).
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB → BGR (OpenCV display/save).
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
