# Skin-Cancer-Classification
🧠 Melanoma Detection Using MobileNetV2 This project is a deep learning-powered web app for classifying skin lesions as either Melanoma or Normal Skin using transfer learning with MobileNetV2.

**Key Features:**
Upload a skin lesion image (JPG/PNG)
Model predicts if it's Melanoma or Normal
Displays prediction confidence
Built with TensorFlow, Keras, OpenCV, and Streamlit
Easy-to-use web interface

**Dataset:**
HAM10000 ("Human Against Machine with 10000 training images") from Kaggle
Total images: ~10,000+ labeled dermatoscopic images

**Labels grouped into:**
Melanoma (mel, bcc, akiec)
Normal Skin (others)

**Tech Stack:**
Python, NumPy, Pandas
TensorFlow/Keras (MobileNetV2)
OpenCV & PIL for image handling
Streamlit for web app interface

**Model Training:**
Transfer learning with pretrained MobileNetV2
Input size: 224x224
Fine-tuned last layers

Achieved ~76.7% validation accuracy in 10 epochs
Balanced training with computed class weights
