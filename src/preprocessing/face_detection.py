import cv2
import numpy as np

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(image_bgr, target_size=(224, 224)):
    """
    Detects face in an image and returns the cropped face.
    
    Args:
        image_bgr (numpy array): Image in BGR format (OpenCV)
        target_size (tuple): Size to resize the face (default 224x224)
    
    Returns:
        face_img (numpy array): Cropped & resized face image
        face_box (tuple): (x, y, w, h) of detected face
    """

    # Convert to grayscale (required for Haar Cascade)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # If no face detected
    if len(faces) == 0:
        return None, None

    # Take the largest detected face
    x, y, w, h = sorted(
        faces, key=lambda box: box[2] * box[3], reverse=True
    )[0]

    # Crop face
    face_img = image_bgr[y:y+h, x:x+w]

    # Resize face
    face_img = cv2.resize(face_img, target_size)

    return face_img, (x, y, w, h)
