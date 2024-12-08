import numpy as np
import pickle
import streamlit as st
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

model_path = 'trained_model_dog_cat.sav'
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Load your custom trained model (assuming it's saved as 'trained_model_dog_cat.sav')
with open('trained_model_dog_cat.sav', 'rb') as file:
    loaded_model = pickle.load(file)

# Image classification function using your custom model
def Img_cls(input_data):
    # Read the image using OpenCV
    input_image = cv2.imread(input_data)

    # Convert BGR (OpenCV format) to RGB (matplotlib format)
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # Display the image using matplotlib
    plt.imshow(input_image_rgb)
    plt.title("Input Image")
    plt.axis("off")  # Hide the axes for better visualization
    plt.show()
    
    # Preprocess the image (resize and normalize)
    input_image_resize = cv2.resize(input_image, (224, 224))  # Resize to match model input
    input_image_scaled = input_image_resize / 255.0  # Normalize pixel values to [0, 1]
    image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])  # Reshape for the model
    
    # Make prediction using the trained model
    input_prediction = loaded_model.predict(image_reshaped)
    
    # Print the raw prediction values (for debugging)
    print("Raw Prediction Values:", input_prediction)
    
    # Get the predicted class label
    input_pred_label = np.argmax(input_prediction)  # Get the index of the highest probability
    
    # Get the accuracy of the prediction
    accuracy = np.max(input_prediction) * 100  # Get the highest confidence percentage
    
    # Display the predicted result and accuracy
    if input_pred_label == 0:
        return f"The image represents a Cat with {accuracy:.2f}% confidence."
    else:
        return f"The image represents a Dog with {accuracy:.2f}% confidence."

# Streamlit frontend function to upload and display images
def main():
    st.title("Image Classification: Dog Vs Cat")
    
    # Image upload for custom model (Dog vs Cat)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image using Streamlit
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Classify the uploaded image using your custom model
        st.write("Classifying...")
        result = Img_cls(img_path)
        st.write(result)

# Run the Streamlit app
if __name__ == "__main__":
    main()
