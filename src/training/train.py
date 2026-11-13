"""Fine-tune ResNet-18 to separate real faces from deepfakes.

Expects:
    data/processed/train/real/*.jpg
    data/processed/train/fake/*.jpg

Usage (from the repository root):
    python -m src.training.train --epochs 5
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.model_architecture import DeepfakeDetector  # noqa: E402

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class FastDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # A corrupt file should not end the run. Step to the next sample,
        # giving up only if every file in the dataset is unreadable.
        for offset in range(len(self.file_paths)):
            current = (idx + offset) % len(self.file_paths)
            try:
                image = Image.open(self.file_paths[current]).convert("RGB")
            except (OSError, ValueError):
                continue
            if self.transform:
                image = self.transform(image)
            return image, self.labels[current]
        raise RuntimeError("No readable images in the dataset.")


def list_images(folder: str, limit: int) -> list[str]:
    """Collect up to `limit` image paths. scandir stays fast on huge folders."""
    paths = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(entry.path)
                if len(paths) >= limit:
                    break
    return paths


def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            preds = (torch.sigmoid(model(images)) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=os.path.join(ROOT_DIR, "data", "processed", "train"))
    parser.add_argument("--output", default=os.path.join(ROOT_DIR, "models", "best_deepfake_model.pth"))
    parser.add_argument("--images-per-class", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}", flush=True)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        NORMALIZE,
    ])

    real_paths = list_images(os.path.join(args.data_dir, "real"), args.images_per_class)
    fake_paths = list_images(os.path.join(args.data_dir, "fake"), args.images_per_class)
    if not real_paths or not fake_paths:
        sys.exit(f"Need images in both {args.data_dir}/real and {args.data_dir}/fake.")

    samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]  # 0=real, 1=fake
    random.shuffle(samples)
    paths, labels = zip(*samples)

    split = int(0.85 * len(paths))
    train_ds = FastDeepfakeDataset(paths[:split], labels[:split], transform=train_transform)
    val_ds = FastDeepfakeDataset(paths[split:], labels[split:], transform=val_transform)

    # num_workers=0 avoids hung worker processes on Windows.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}", flush=True)

    model = DeepfakeDetector(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch} | batch {step}/{len(train_loader)} | loss {loss.item():.4f}", flush=True)

        val_acc = accuracy(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | loss {running_loss / len(train_loader):.4f} "
            f"| val acc {val_acc:.2f}%",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"Saved new best model to {args.output}", flush=True)


if __name__ == "__main__":
    main()
