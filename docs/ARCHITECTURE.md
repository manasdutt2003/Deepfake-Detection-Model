# Architecture

## Pipeline

```
image -> detect_and_crop_face (Haar cascade, largest face, 224x224 BGR crop)
      -> predict_face: validate -> BGR to RGB -> resize/normalise (ImageNet stats)
      -> DeepfakeDetector (ResNet-18, one logit) -> sigmoid -> p(fake)
      -> classify: REAL / UNCERTAIN / DEEPFAKE
```

## Modules

| Module | Responsibility |
| --- | --- |
| `src/preprocessing/face_detection.py` | Finds faces with OpenCV's Haar cascade and returns the largest one, cropped and resized. Returns `(None, None)` when no face is found. |
| `models/model_architecture.py` | `DeepfakeDetector`: torchvision ResNet-18 whose final layer is replaced with a single-output linear layer. |
| `src/inference/predict.py` | Loads the checkpoint once (`lru_cache`), validates input, runs the model, maps the probability to a label. Also provides the CLI. |
| `src/training/train.py`, `evaluate.py` | Training and evaluation scripts. |
| `app/streamlit_app.py` | Login, image upload, result display and metrics view. |

## Design decisions

- **Raw logits from the model, sigmoid at inference.** This keeps the model compatible with a binary cross-entropy loss with logits during training.
- **Three-way verdict.** Probabilities between the two thresholds are reported as UNCERTAIN rather than forced into a label. The thresholds are `0.35` and `0.65` by default.
- **Model cached per path.** `load_trained_model` is cached so the Streamlit app does not reload the checkpoint for each upload.
- **`weights_only=True` when loading checkpoints**, so loading a checkpoint cannot execute arbitrary pickled code.

## Configuration

| Variable | Default | Effect |
| --- | --- | --- |
| `DEEPFAKE_MODEL_PATH` | `models/best_deepfake_model.pth` | Checkpoint used for inference |
| `DEEPFAKE_FAKE_THRESHOLD` | `0.65` | Above this the verdict is DEEPFAKE |
| `DEEPFAKE_REAL_THRESHOLD` | `0.35` | Below this the verdict is REAL |
| `APP_USERNAME`, `APP_PASSWORD` | `admin`, `admin` | Login for the Streamlit app (demo defaults) |

Thresholds must be numbers between 0 and 1, and the real threshold must not exceed the fake threshold; otherwise startup fails with an explanatory error.

## Error handling

- Unreadable uploads are reported in the UI; no face detected is reported without running the model.
- `predict_face` raises `ValueError` for non-image arrays and `FileNotFoundError` for a missing checkpoint.

## Known limitations

See the Limitations section of the README. The Haar cascade is a simple detector and the model was trained on GAN-generated faces only.
