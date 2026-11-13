# Deepfake Detection Model

Detects manipulated face images. A Haar-cascade detector crops the largest face
in a photo, a fine-tuned ResNet-18 scores it, and a Streamlit app shows the
verdict with the underlying probability.

```
image -> face detection (OpenCV Haar cascade) -> 224x224 crop
      -> ResNet-18 (single logit) -> sigmoid -> REAL / UNCERTAIN / DEEPFAKE
```

## How a verdict is reached

The model outputs one number: the probability that the face is fake.

| Probability of deepfake | Verdict |
| --- | --- |
| above 0.65 | DEEPFAKE |
| 0.35 to 0.65 | UNCERTAIN |
| below 0.35 | REAL |

## Run the app

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The demo login is `admin` / `admin`. Set `APP_USERNAME` and `APP_PASSWORD`
(environment variables or Streamlit secrets) before exposing the app publicly.

## Classify one image from the command line

```bash
python -m src.inference.predict path/to/photo.jpg
```

## Train

Arrange the data as:

```
data/processed/train/real/*.jpg
data/processed/train/fake/*.jpg
```

```bash
python -m src.training.train --epochs 5 --images-per-class 15000
```

The best checkpoint by validation accuracy is written to
`models/best_deepfake_model.pth`.

## Evaluate

```
data/processed/test/real/*.jpg
data/processed/test/fake/*.jpg
```

```bash
python -m src.training.evaluate --images-per-class 500 --save-json models/metrics.json
```

This prints accuracy, precision, recall, F1 and the confusion matrix (positive
class = fake). When `models/metrics.json` exists the app displays it; otherwise
it shows the figures recorded at training time.

## Project layout

| Path | Purpose |
| --- | --- |
| `app/streamlit_app.py` | Web interface |
| `models/model_architecture.py` | ResNet-18 with a one-logit head |
| `models/best_deepfake_model.pth` | Checkpoint used by the app |
| `models/trained_model.pth` | Earlier checkpoint, kept for comparison |
| `src/preprocessing/face_detection.py` | Face detection and cropping |
| `src/inference/predict.py` | Model loading and prediction |
| `src/training/train.py`, `evaluate.py` | Training and evaluation scripts |

## Limitations

- Trained on GAN-generated faces. Diffusion-model images, face swaps in video
  and heavily compressed images may not be detected.
- Only the largest face in the image is analysed, and the Haar cascade misses
  profile or occluded faces.
- A verdict is a statistical estimate, not proof. Do not use it as the only
  basis for a decision about a real person.
