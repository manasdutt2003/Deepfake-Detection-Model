"""Single-image inference for the deepfake detector.

Usage:
    python -m src.inference.predict path/to/image.jpg
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache

import cv2
import torch
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.model_architecture import DeepfakeDetector  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_deepfake_model.pth")

# Probability of "fake" above/below which the model commits to a label.
FAKE_THRESHOLD = 0.65
REAL_THRESHOLD = 0.35

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@lru_cache(maxsize=2)
def load_trained_model(model_path: str = DEFAULT_MODEL_PATH) -> DeepfakeDetector:
    """Load the checkpoint once and reuse it for every prediction."""
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
        face_bgr: Face crop as an OpenCV BGR array.

    Returns:
        (label, fake_probability) where fake_probability is the sigmoid
        output in [0, 1]; values near 1 mean the face is likely manipulated.
    """
    model = load_trained_model(model_path)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(face_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fake_probability = torch.sigmoid(model(tensor)).item()

    return classify(fake_probability), fake_probability


def main() -> None:
    from src.preprocessing.face_detection import detect_and_crop_face

    parser = argparse.ArgumentParser(description="Classify one image as real or deepfake")
    parser.add_argument("image", help="Path to an image containing a face")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Could not read image: {args.image}")

    face, _ = detect_and_crop_face(image)
    if face is None:
        sys.exit("No face detected in the image.")

    label, fake_probability = predict_face(face, args.model_path)
    print(f"Result: {label}")
    print(f"Probability of deepfake: {fake_probability:.4f}")


if __name__ == "__main__":
    main()
