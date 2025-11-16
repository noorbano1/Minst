import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# Load the trained model
model = load_model("mnist_cnn.h5")

st.title("MNIST Handwritten Digit Recognition")
st.write("Upload a grayscale image of size 28x28 and the model will predict the digit.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and process the image
    img = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors if needed (white background)
    img = img.resize((28,28))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Convert to array
    img_array = np.array(img).astype('float32') / 255
    img_array = img_array.reshape(1,28,28,1)  # Reshape for CNN

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100

    st.write(f"Predicted Digit: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
