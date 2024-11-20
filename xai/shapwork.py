import shap
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

def apply_shap_integrated_gradients(image_path, model, target_size=(224, 224)):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    # Normalize the image
    img_array_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array_expanded)

    # Use SHAP's GradientExplainer for explaining the model (works similarly to Integrated Gradients)
    explainer = shap.GradientExplainer(model, img_array_preprocessed)
    shap_values = explainer.shap_values(img_array_preprocessed)

    # Visualize SHAP values
    shap.image_plot(shap_values, img_array)

    # Optionally, save the SHAP explanation
    save_shap_explanation(shap_values[0], img)

def save_shap_explanation(shap_values, img):
    """
    Save the SHAP explanation as an image.
    """
    shap_values = np.mean(shap_values, axis=-1)  # Take the mean over the channels (RGB)
    shap_values = np.uint8(shap_values * 255)  # Scale to [0, 255] for image display
    
    # Convert SHAP values to an image and save
    from PIL import Image
    shap_img = Image.fromarray(shap_values)
    shap_img.save("shap_explanation.png")
    print("SHAP explanation saved as 'shap_explanation.png'.")

# Test the function
if __name__ == "__main__":
    image_path = "C:/Users/Medha Agarwal/Desktop/GANs/augmented_mri_images/augmented_mri_0.png"  # Path to the image you want to explain
    model = tf.keras.applications.ResNet50(weights='imagenet')  # Your model

    apply_shap_integrated_gradients(image_path, model)
