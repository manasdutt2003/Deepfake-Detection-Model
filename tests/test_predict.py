"""Tests that run without the model checkpoint.

Run from the repository root: python -m pytest tests -q
"""

import numpy as np
import pytest
import torch

from src.inference import predict
from src.preprocessing.face_detection import detect_and_crop_face


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.0, "REAL"),
        (0.34, "REAL"),
        (0.35, "UNCERTAIN"),  # the thresholds themselves are not committed labels
        (0.5, "UNCERTAIN"),
        (0.65, "UNCERTAIN"),
        (0.66, "DEEPFAKE"),
        (1.0, "DEEPFAKE"),
    ],
)
def test_classify_default_thresholds(probability, expected):
    # Assumes the default thresholds (no DEEPFAKE_* overrides in the environment).
    if (predict.REAL_THRESHOLD, predict.FAKE_THRESHOLD) != (0.35, 0.65):
        pytest.skip("threshold environment variables are set")
    assert predict.classify(probability) == expected


@pytest.mark.parametrize(
    "bad_input",
    [None, np.array([]), np.zeros((10, 10)), np.zeros((10, 10, 4), dtype=np.uint8)],
)
def test_predict_face_rejects_bad_input(bad_input):
    with pytest.raises(ValueError):
        predict.predict_face(bad_input)


class _StubModel:
    def __init__(self, logit):
        self.logit = logit

    def __call__(self, tensor):
        return torch.tensor([[self.logit]])


def test_predict_face_uses_sigmoid_of_logit(monkeypatch):
    monkeypatch.setattr(predict, "load_trained_model", lambda path=None: _StubModel(10.0))
    face = np.zeros((224, 224, 3), dtype=np.uint8)
    label, probability = predict.predict_face(face)
    assert probability == pytest.approx(torch.sigmoid(torch.tensor(10.0)).item())
    assert label == "DEEPFAKE"


def test_missing_checkpoint_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load_trained_model(str(tmp_path / "missing.pth"))


def test_blank_image_has_no_face():
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    face, box = detect_and_crop_face(blank)
    assert face is None and box is None
