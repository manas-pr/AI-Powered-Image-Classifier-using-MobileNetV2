import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def predict(image_file, model):
    img = Image.open(image_file).resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    return decoded_predictions

def main():
    st.title("🖼️ AI-Powered Image Classifier")
    st.write("Upload an image, and the AI will classify it!")
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        st.write("Classifying...")
        predictions = predict(uploaded_file, model)
        
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"**{label}**: {score*100:.2f}%")

if __name__ == "__main__":
    main()
