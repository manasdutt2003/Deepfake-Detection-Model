"""Evaluate a trained checkpoint on a balanced sample of the test set.

Expects an ImageFolder layout whose class folders sort as fake, real:
    data/processed/test/fake/*.jpg
    data/processed/test/real/*.jpg

Usage (from the repository root):
    python -m src.training.evaluate --images-per-class 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.model_architecture import DeepfakeDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.path.join(ROOT_DIR, "data", "processed", "test"))
    parser.add_argument("--model-path", default=os.path.join(ROOT_DIR, "models", "best_deepfake_model.pth"))
    parser.add_argument("--images-per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", help="Write the metrics to this JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    print(f"Test images found: {len(dataset)} | classes: {dataset.class_to_idx}", flush=True)

    # The model was trained with 0=real and 1=fake. ImageFolder numbers the
    # classes alphabetically, so translate its indices instead of assuming
    # they match.
    if "fake" not in dataset.class_to_idx or "real" not in dataset.class_to_idx:
        sys.exit(f"Expected 'fake' and 'real' folders in {args.data_dir}.")
    fake_index = dataset.class_to_idx["fake"]

    by_class: dict[int, list[int]] = {index: [] for index in dataset.class_to_idx.values()}
    for position, (_, class_index) in enumerate(dataset.samples):
        by_class[class_index].append(position)

    selected: list[int] = []
    for positions in by_class.values():
        rng.shuffle(positions)
        selected.extend(positions[: args.images_per_class])

    loader = DataLoader(Subset(dataset, selected), batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Evaluating on {len(selected)} images", flush=True)

    model = DeepfakeDetector(pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device).eval()

    predictions: list[int] = []
    targets: list[int] = []
    with torch.no_grad():
        for images, class_indices in loader:
            # The model returns logits, so threshold the sigmoid at 0.5.
            probabilities = torch.sigmoid(model(images.to(device)))
            predictions.extend((probabilities >= 0.5).int().cpu().numpy().ravel().tolist())
            targets.extend((class_indices == fake_index).int().numpy().tolist())

    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "confusion_matrix": matrix.tolist(),
        "labels": ["real", "fake"],
        "images_evaluated": len(targets),
    }

    print("\nEvaluation results (positive class = fake)")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"{name:<10}: {metrics[name]:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted; order real, fake):")
    print(matrix)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"\nSaved metrics to {args.save_json}")


if __name__ == "__main__":
    main()
