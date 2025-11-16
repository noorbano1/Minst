import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Page config
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🎨",
    layout="centered"
)

# -----------------------------
# Colorful Title
# -----------------------------
st.markdown("<h1 style='color:#4B0082; text-align:center;'>🎨 MNIST Handwritten Digit Recognition 🎨</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:#FF4500; text-align:center;'>Upload a 28x28 grayscale image and see the predicted digit!</h4>", unsafe_allow_html=True)
st.markdown("---")  # horizontal line

# Load the trained model
model = load_model("mnist_cnn.h5")

# -----------------------------
# File uploader section
# -----------------------------
st.markdown("<h3 style='color:#008B8B;'>Step 1: Upload Your Image</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and process the image
    img = Image.open(uploaded_file).convert("L")  # Grayscale
    img = ImageOps.invert(img)  # Invert colors if needed
    img = img.resize((28,28))
    
    st.markdown("<h4 style='color:#2E8B57;'>Uploaded Image:</h4>", unsafe_allow_html=True)
    st.image(img, caption='Your Image', use_column_width=True)

    # Convert to array
    img_array = np.array(img).astype('float32') / 255
    img_array = img_array.reshape(1,28,28,1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    st.markdown("<h3 style='color:#FF1493;'>Prediction Result</h3>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color:#1E90FF;'>Predicted Digit: {predicted_class}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:#DAA520;'>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<p style='color:#696969; text-align:center;'>Developed with ❤️ using Streamlit and TensorFlow</p>", unsafe_allow_html=True)
