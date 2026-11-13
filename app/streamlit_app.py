"""Streamlit front end for the deepfake detector."""

import hmac
import json
import os
import sys

# Torch and OpenCV can each load their own OpenMP runtime on Windows, which
# aborts the process (OMP error #15) unless duplicates are allowed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from src.inference.predict import (  # noqa: E402
    FAKE_THRESHOLD,
    REAL_THRESHOLD,
    predict_face,
)
from src.preprocessing.face_detection import detect_and_crop_face  # noqa: E402

METRICS_PATH = os.path.join(ROOT_DIR, "models", "metrics.json")

# Figures recorded when the model was trained. They are replaced by
# models/metrics.json when that file exists (see src/training/evaluate.py).
REPORTED_METRICS = {
    "accuracy": 0.912,
    "precision": 0.895,
    "recall": 0.931,
    "f1": 0.912,
    "confusion_matrix": [[3650, 350], [280, 3720]],
}

st.set_page_config(
    page_title="Deepfake Detection | Forensic AI",
    page_icon="🛡️",
    layout="wide",
)


def get_setting(name: str, default: str) -> str:
    """Read a setting from Streamlit secrets, then the environment."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass  # no secrets.toml: normal for local runs
    return os.getenv(name, default)


APP_USERNAME = get_setting("APP_USERNAME", "admin")
APP_PASSWORD = get_setting("APP_PASSWORD", "admin")
USING_DEFAULT_LOGIN = (APP_USERNAME, APP_PASSWORD) == ("admin", "admin")


def credentials_match(username: str, password: str) -> bool:
    # compare_digest avoids leaking how many characters matched via timing.
    return hmac.compare_digest(username.encode(), APP_USERNAME.encode()) and hmac.compare_digest(
        password.encode(), APP_PASSWORD.encode()
    )


def load_metrics() -> tuple[dict, str]:
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, encoding="utf-8") as handle:
                return json.load(handle), "Measured by src/training/evaluate.py"
        except (OSError, ValueError):
            pass
    return REPORTED_METRICS, "Reported at training time"


def logout() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def login_page() -> None:
    _, col_main, _ = st.columns([1, 1.2, 1])
    with col_main:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align: center;">
                <h1 style='color: #1E3A8A; font-size: 24px;'>🛡️ Deepfake detection model</h1>
                <p style='color: #6B7280;'>AI-powered detection of manipulated facial images</p>
                <hr>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if credentials_match(username, password):
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

        if USING_DEFAULT_LOGIN:
            st.caption(
                "Demo login: `admin` / `admin`. Set `APP_USERNAME` and "
                "`APP_PASSWORD` to change it."
            )


def performance_section() -> None:
    metrics, source = load_metrics()
    st.divider()
    cols = st.columns(4)
    for col, (label, key) in zip(
        cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1 Score", "f1")]
    ):
        col.metric(label, f"{metrics[key] * 100:.1f}%")
    st.caption(source)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        np.array(metrics["confusion_matrix"]),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)
    st.divider()


def analysis_section() -> None:
    st.subheader("🔍 Image Evaluation")
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.write("### 📥 Source Image")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        image = None
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except (UnidentifiedImageError, OSError):
                st.error("That file could not be read as an image.")
            else:
                st.image(image, caption="Input for Analysis", use_container_width=True)

    with col_result:
        st.write("### 🔬 Analysis Result")
        if image is None:
            st.info("Awaiting image upload...")
            return

        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        with st.spinner("Processing..."):
            face, _ = detect_and_crop_face(image_bgr)

        if face is None:
            st.error("No face detected. Try a clearer, front-facing photo.")
            return

        try:
            label, fake_probability = predict_face(face)
        except Exception as error:
            st.error(f"The model could not be loaded or run: {error}")
            return

        roi_col, verdict_col = st.columns([1, 1])
        with roi_col:
            st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption="Detected Face (ROI)", width=180)

        with verdict_col:
            if label == "DEEPFAKE":
                confidence, delta_color = fake_probability, "inverse"
            elif label == "REAL":
                confidence, delta_color = 1 - fake_probability, "normal"
            else:
                confidence, delta_color = None, "off"
            delta = f"Confidence: {confidence:.1%}" if confidence is not None else "Between thresholds"
            st.metric(label="Decision", value=label, delta=delta, delta_color=delta_color)
            st.progress(fake_probability, text=f"Probability of deepfake: {fake_probability:.1%}")

        with st.expander("📝 Technical Data"):
            st.write(f"**Probability of deepfake (sigmoid output):** `{fake_probability:.4f}`")
            st.write(
                f"**Decision thresholds:** above `{FAKE_THRESHOLD}` is DEEPFAKE, "
                f"below `{REAL_THRESHOLD}` is REAL, otherwise UNCERTAIN"
            )
            st.write("**Face detector:** OpenCV Haar cascade (largest face is used)")


def main_app() -> None:
    with st.sidebar:
        st.header("👤 Session Info")
        st.write(f"**Account:** {APP_USERNAME}")
        if st.button("Secure Logout", type="primary"):
            logout()
        st.divider()
        st.info("**Model:** ResNet-18 (binary head)\n\n**Input:** 224×224 face crop")

    st.title("🛡️ Deepfake detection model")
    st.subheader("AI-powered detection of manipulated facial images")

    if st.toggle("📊 View Model Performance & Confusion Matrix", value=False):
        performance_section()

    analysis_section()


if st.session_state.get("logged_in"):
    main_app()
else:
    login_page()
