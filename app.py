import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# MobileNet imports (IMPORTANT)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Animal Detection CNN",
    page_icon="🐾",
    layout="centered"
)

# =============================
# LOAD MODEL & CLASSES
# =============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model.keras",   # or "model.h5"
        compile=False,
        custom_objects={
            "MobileNetV2": MobileNetV2
        }
    )
    
@st.cache_data
def load_classes():
    with open("class_names.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
class_names = load_classes()

IMG_SIZE = 224

# =============================
# ITALIAN → ENGLISH LABEL MAP
# =============================
label_map = {
    "cane": "Dog",
    "gatto": "Cat",
    "cavallo": "Horse",
    "elefante": "Elephant",
    "farfalla": "Butterfly",
    "gallina": "Chicken",
    "mucca": "Cow",
    "pecora": "Sheep",
    "ragno": "Spider",
    "scoiattolo": "Squirrel"
}

# =============================
# CUSTOM CSS
# =============================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    font-size: 15px;
    color: #94a3b8;
    margin-bottom: 25px;
}
.card {
    background: #020617;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
.result {
    font-size: 22px;
    font-weight: bold;
    margin-top: 15px;
}
.conf {
    font-size: 16px;
    color: #22c55e;
}
</style>
""", unsafe_allow_html=True)

# =============================
# TITLE
# =============================
st.markdown('<div class="title">🐾 Animal Detection CNN</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an image and the model will predict the animal</div>',
    unsafe_allow_html=True
)

# =============================
# MAIN CARD
# =============================
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload an animal image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # =============================
    # PREPROCESS IMAGE (CORRECT FOR MOBILENET)
    # =============================
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # =============================
    # PREDICT
    # =============================
    preds = model.predict(img_array, verbose=0)
    confidence = float(np.max(preds))

    raw_class = class_names[np.argmax(preds)]
    predicted_class = label_map.get(raw_class, raw_class)

    # =============================
    # SHOW RESULT
    # =============================
    st.markdown(
        f'<div class="result">Prediction: 🐾 <span style="color:#38bdf8">{predicted_class}</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="conf">Confidence: {confidence * 100:.2f}%</div>',
        unsafe_allow_html=True
    )
else:
    st.info("📸 Please upload an image")

st.markdown('</div>', unsafe_allow_html=True)
