import torch
import cv2
import numpy as np
import sys
import os
from torchvision import transforms

# ---------------- PATH & IMPORT FIX ----------------
# Force add the project root to system path so we can find 'models'
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels: src/inference -> src -> root
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)

from models.model_architecture import DeepfakeDetector

# ---------------- DEVICE ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
def load_trained_model(model_path=None):
    # Use absolute path for the model file to avoid file-not-found errors
    if model_path is None:
        model_path = os.path.join(root_dir, "models", "best_deepfake_model.pth")
    
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

# Initialize model
try:
    model = load_trained_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- PREDICTION FUNCTION ----------------
def predict_face(face_bgr):
    if model is None:
        return "ERROR: Model not loaded", 0.0

    # 1. Convert BGR (OpenCV) to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # 2. Preprocess
    face_tensor = transform(face_rgb).unsqueeze(0).to(DEVICE)

    # 3. Inference
    with torch.no_grad():
        logits = model(face_tensor)
        score = torch.sigmoid(logits).item()

    # 4. Logic
    if score > 0.65:
        label = "DEEPFAKE"
        confidence = score * 100
    elif score < 0.35:
        label = "REAL"
        confidence = (1 - score) * 100
    else:
        label = "UNCERTAIN"
        confidence = score * 100 

    return label, confidence

if __name__ == "__main__":
    test_image_path = "path_to_your_test_image.jpg"
    image = cv2.imread(test_image_path)
    
    if image is not None:
        result_label, result_conf = predict_face(image)
        print("-" * 30)
        print(f"RESULT: {result_label}")
        print(f"CONFIDENCE: {result_conf:.2f}%")
        print("-" * 30)
    else:
        print("❌ Could not load image. Check path!")