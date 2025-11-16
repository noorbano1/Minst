from fpdf import FPDF

# Assignment-ready README content
readme_text = """
MNIST Handwritten Digit Recognition Web App

Overview:
A Streamlit web application that uses a CNN trained on the MNIST dataset 
to recognize handwritten digits (0–9). Users can upload a 28x28 grayscale image, 
and the app predicts the digit with a confidence score.

Features:
- Upload handwritten digits for real-time prediction
- Displays predicted digit and confidence
- Easy-to-use web interface with Streamlit
- Fully trained CNN achieving high test accuracy (~99%)

Installation:
1. Clone the repository:
   git clone <repo-link>
   cd <repo-folder>
2. Install dependencies:
   pip install -r requirements.txt
3. Ensure the trained model mnist_cnn.h5 is in the project folder.

Usage:
Run the Streamlit app:
   streamlit run app.py
Upload a 28x28 grayscale image, and view the predicted digit and confidence.

CNN Architecture:
- Conv2D layers: 2 layers (32 & 64 filters, 3x3 kernel, ReLU)
- MaxPooling2D layers: 2 layers (2x2)
- Flatten layer
- Dense layer: 128 neurons, ReLU
- Dropout: 0.3
- Output layer: 10 neurons, Softmax

Results:
- Test accuracy: ~99%
- Confusion matrix and training curves show strong performance
- Model performs well with minimal overfitting

Dependencies:
- tensorflow, streamlit, numpy, pillow, matplotlib, pandas, seaborn

Future Improvements:
- Add a drawing canvas to draw digits directly in the app
- Automatic preprocessing of color images
- Deploy on Streamlit Cloud or Heroku
"""
