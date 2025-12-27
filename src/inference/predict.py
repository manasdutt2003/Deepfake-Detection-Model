import torch
import cv2
import numpy as np
from torchvision import transforms
from models.model_architecture import DeepfakeDetector

# ---------------- DEVICE ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
def load_trained_model(model_path="models/best_deepfake_model.pth"):
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_trained_model()

# ---------------- TRANSFORMS ----------------
# Standard ResNet normalization used in training
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
    """
    Inputs: 
        face_bgr: Image array from cv2.imread
    Returns:
        label (str): Real / Deepfake / Uncertain
        confidence (float): The percentage of certainty
    """
    # 1. Convert BGR (OpenCV) to RGB (what the model expects)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # 2. Preprocess
    face_tensor = transform(face_rgb).unsqueeze(0).to(DEVICE)

    # 3. Inference
    with torch.no_grad():
        logits = model(face_tensor)
        # Sigmoid turns the raw logit into a probability between 0 and 1
        # Closer to 1 = Deepfake | Closer to 0 = Real
        score = torch.sigmoid(logits).item()

    # 4. Logic (Corrected for 1 = Deepfake)
    if score > 0.65:
        label = "DEEPFAKE"
        confidence = score * 100
    elif score < 0.35:
        label = "REAL"
        confidence = (1 - score) * 100
    else:
        label = "UNCERTAIN"
        confidence = score * 100 # or 50% midpoint

    return label, confidence

# ---------------- EXAMPLE USAGE ----------------
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