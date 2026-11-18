"""Single-image inference for the deepfake detector.

Usage:
    python -m src.inference.predict path/to/image.jpg

Environment variables (all optional):
    DEEPFAKE_MODEL_PATH       Path to the checkpoint to load.
    DEEPFAKE_FAKE_THRESHOLD   Probability above which the label is DEEPFAKE (default 0.65).
    DEEPFAKE_REAL_THRESHOLD   Probability below which the label is REAL (default 0.35).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from functools import lru_cache

import cv2
import numpy as np
import torch
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.model_architecture import DeepfakeDetector  # noqa: E402

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = os.environ.get(
    "DEEPFAKE_MODEL_PATH",
    os.path.join(ROOT_DIR, "models", "best_deepfake_model.pth"),
)


def _read_threshold(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


# Probability of "fake" above/below which the model commits to a label.
FAKE_THRESHOLD = _read_threshold("DEEPFAKE_FAKE_THRESHOLD", 0.65)
REAL_THRESHOLD = _read_threshold("DEEPFAKE_REAL_THRESHOLD", 0.35)
if REAL_THRESHOLD > FAKE_THRESHOLD:
    raise ValueError(
        f"DEEPFAKE_REAL_THRESHOLD ({REAL_THRESHOLD}) must not exceed "
        f"DEEPFAKE_FAKE_THRESHOLD ({FAKE_THRESHOLD})"
    )

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=2)
def load_trained_model(model_path: str = DEFAULT_MODEL_PATH) -> DeepfakeDetector:
    """Load the checkpoint once and reuse it for every prediction."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    logger.info("Loading checkpoint %s on %s", model_path, DEVICE)
    model = DeepfakeDetector(pretrained=False)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def classify(fake_probability: float) -> str:
    if fake_probability > FAKE_THRESHOLD:
        return "DEEPFAKE"
    if fake_probability < REAL_THRESHOLD:
        return "REAL"
    return "UNCERTAIN"


def predict_face(face_bgr, model_path: str = DEFAULT_MODEL_PATH) -> tuple[str, float]:
    """Classify a cropped face.

    Args:
        face_bgr: Face crop as an OpenCV BGR array (height x width x 3).

    Returns:
        (label, fake_probability) where fake_probability is the sigmoid
        output in [0, 1]; values near 1 mean the face is likely manipulated.

    Raises:
        ValueError: if the input is not a non-empty 3-channel image array.
        FileNotFoundError: if the checkpoint does not exist.
    """
    if not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
        raise ValueError("face_bgr must be a non-empty numpy array")
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        raise ValueError(
            f"face_bgr must have shape (height, width, 3), got {face_bgr.shape}"
        )

    model = load_trained_model(model_path)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(face_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_probability = torch.sigmoid(model(tensor)).item()

    label = classify(fake_probability)
    logger.info("Prediction: %s (p_fake=%.4f)", label, fake_probability)
    return label, fake_probability


def main() -> None:
    from src.preprocessing.face_detection import detect_and_crop_face

    parser = argparse.ArgumentParser(description="Classify one image as real or deepfake")
    parser.add_argument("image", help="Path to an image containing a face")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--verbose", action="store_true", help="Enable info logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Could not read image: {args.image}")

    face, _ = detect_and_crop_face(image)
    if face is None:
        sys.exit("No face detected in the image.")

    try:
        label, fake_probability = predict_face(face, args.model_path)
    except FileNotFoundError as error:
        sys.exit(str(error))
    print(f"Result: {label}")
    print(f"Probability of deepfake: {fake_probability:.4f}")


if __name__ == "__main__":
    main()
