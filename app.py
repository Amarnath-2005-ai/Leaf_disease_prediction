import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

MODEL_PATH = "plant_disease_recog_model_pwp.keras"
FILE_ID = "1HJBCQTyOrTKgBNJ7gH-H9lB37df_81-7"
GDRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = ['Healthy', 'Bacterial Spot', 'Leaf Rust']  # Replace with your real labels

st.title("🌿 Leaf Disease Recognition")
st.write("Upload a leaf image to predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))  # Match your model's input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Prediction: **{predicted_class}**")
