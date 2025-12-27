import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
import random
from PIL import Image
from models.model_architecture import DeepfakeDetector

class FastDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.file_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # If a file is corrupt, return the first item instead of crashing
            return self.__getitem__(0)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {DEVICE}", flush=True)

    # --- SCALE CONFIG ---
    IMAGES_PER_CLASS = 15000     # Safely increased (Total 30,000 images)
    BATCH_SIZE = 32              # Increase to 64 if you have a strong GPU
    EPOCHS = 5                   
    LEARNING_RATE = 1e-4

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- FAST DIRECT INDEXING ---
    root_dir = "data/processed/train"
    real_dir = os.path.join(root_dir, "real")
    fake_dir = os.path.join(root_dir, "fake")

    print(f"📂 Selecting {IMAGES_PER_CLASS*2} total images...", flush=True)
    
    # Use os.scandir for much faster performance on large folders
    def get_paths(folder, count):
        paths = []
        with os.scandir(folder) as it:
            for entry in it:
                if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(entry.path)
                    if len(paths) >= count:
                        break
        return paths

    real_paths = get_paths(real_dir, IMAGES_PER_CLASS)
    fake_paths = get_paths(fake_dir, IMAGES_PER_CLASS)

    all_paths = real_paths + fake_paths
    all_labels = [0] * len(real_paths) + [1] * len(fake_paths) # 0=Real, 1=Fake

    # Shuffle
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)

    split = int(0.85 * len(all_paths))
    train_ds = FastDeepfakeDataset(all_paths[:split], all_labels[:split], transform=transform)
    val_ds = FastDeepfakeDataset(all_paths[split:], all_labels[split:], transform=transform)

    # num_workers=0 is still recommended for Windows to avoid 'stuck' processes
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"✅ Data Ready | Train: {len(train_ds)} | Val: {len(val_ds)}", flush=True)

    # --- INITIALIZE MODEL ---
    model = DeepfakeDetector().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- TRAINING LOOP ---
    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        print(f"\n🔁 Epoch {epoch+1}/{EPOCHS}", flush=True)

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if i % 50 == 0:
                print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
                preds = (torch.sigmoid(model(images)) > 0.5).float()
                correct += (preds == labels).sum().item()

        val_acc = 100 * correct / len(val_ds)
        print(f"✨ Epoch {epoch+1} Summary | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_deepfake_model.pth")
            print("💾 New Best Model Saved!", flush=True)

if __name__ == "__main__":
    main()