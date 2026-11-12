# app.py
"""Simple Streamlit dashboard for the Agri Quality Classifier.
Upload a fruit image (Apple, Banana, Grape, Guava, Orange) – healthy or rotten.
The trained model is loaded once and used to predict the class with confidence.
"""

import pathlib
import shutil
import streamlit as st
from src.inference import predict_image, load_checkpoint
from config import InferenceConfig

st.set_page_config(page_title="Agri Quality Classifier", layout="centered")
st.title("🍎 Agri Quality Classifier")
st.markdown("Detect whether an uploaded fruit image is **healthy** or **rotten** for the selected five fruits.")

# -----------------------------------------------------------------------------
# Load model (cached – executed only once per session)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model_and_mapping():
    # The inference module decides CPU/GPU automatically.
    model, class_idx_to_name = load_checkpoint()
    return model, class_idx_to_name

model, class_idx_to_name = get_model_and_mapping()

# -----------------------------------------------------------------------------
# File uploader UI
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded image to a temporary location for PIL to open.
    tmp_dir = pathlib.Path("./tmp")
    tmp_dir.mkdir(exist_ok=True)
    img_path = tmp_dir / uploaded_file.name
    img_path.write_bytes(uploaded_file.getvalue())

    st.image(str(img_path), caption="Uploaded image", use_column_width=True)

    # Run inference
    cfg = InferenceConfig()
    result = predict_image(img_path, cfg=cfg)

    st.subheader("Prediction")
    st.write(f"**Class:** {result['predicted_class']}")
    st.write(f"**Confidence:** {result['confidence']:.2%}")
    if "warning" in result:
        st.warning(result["warning"])

    st.subheader("Top‑3 predictions")
    for cls, prob in result["top3"]:
        st.write(f"{cls}: {prob:.2%}")

    # Optional clean‑up of the temporary file
    try:
        img_path.unlink()
        # Remove the tmp folder if empty
        if not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    except Exception:
        pass
else:
    st.info("Upload an image to get a prediction.")

# -----------------------------------------------------------------------------
# Sidebar information
# -----------------------------------------------------------------------------
st.sidebar.title("About")
st.sidebar.info(
    "The model was trained on the Fruit & Vegetable Disease dataset using a MobileNet‑V3‑Small backbone. "
    "It classifies five fruits (Apple, Banana, Grape, Guava, Orange) into healthy or rotten categories."
)
