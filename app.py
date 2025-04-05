import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Title
st.title("🧠 Melanoma Classifier (MobileNetV2)")
st.markdown("Upload a skin image to check if it's **Melanoma** or **Normal Skin**.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("melanoma_classifier_fixed.h5")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:  # RGBA → RGB
        img_array = img_array[..., :3]

    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "Melanoma" if prediction >= 0.5 else "Normal Skin"
    confidence = float(prediction) if prediction >= 0.5 else 1 - float(prediction)

    # Result
    st.subheader("🩺 Prediction:")
    st.write(f"**{label}**")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
