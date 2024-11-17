import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
from lime import lime_image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load base model (ResNet50) and add custom layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Grad-CAM function
def get_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), predictions.numpy()

# Function to prepare the model for LIME
def predict_fn(images):
    images = np.array(images)
    images = preprocess_input(images)
    return model.predict(images)

# Function to visualize Grad-CAM and LIME with an explanation

def apply_gradcam_and_lime(image_path, model, target_size=(128, 128), opacity=0.4):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = preprocess_input(img_array_expanded)

    # Specify the last conv layer for ResNet50
    last_conv_layer_name = 'conv5_block3_out'
    heatmap, predictions = get_gradcam_heatmap(model, img_array_preprocessed, last_conv_layer_name)

    # Prediction probabilities
    class_names = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']  # Replace with actual class names
    confidences = {class_names[i]: predictions[0][i] for i in range(len(class_names))}

    # LIME Explanation
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array.astype('double'), predict_fn, top_labels=4, hide_color=0, num_samples=1000)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=80)  # Smaller figure size and dpi for better image fit

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Heatmap
    axes[0, 1].imshow(heatmap, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap')
    axes[0, 1].axis('off')

    # Superimposed Image
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    heatmap_img = np.uint8(255 * heatmap)
    
    # Use jet colormap (Updated for compatibility with newer versions of Matplotlib)
    jet = plt.cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_img]
    
    # Resize and convert
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(target_size)
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose
    superimposed = jet_heatmap * opacity + img_array
    superimposed = tf.keras.preprocessing.image.array_to_img(superimposed)

    axes[0, 2].imshow(superimposed)
    axes[0, 2].set_title('Superimposed Image')
    axes[0, 2].axis('off')

    # Bar chart for confidence scores
    axes[1, 0].barh(list(confidences.keys()), list(confidences.values()), color='skyblue')
    axes[1, 0].set_title('Prediction Confidence')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_xlim(0, 1)  # Ensure full visibility of the x-axis

    # LIME Explanation Overlay
    lime_overlay_result = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=True)

    # Check if the lime_overlay_result is not None before proceeding
    if lime_overlay_result:
        # Convert the LIME image to the appropriate dtype for display
        lime_explanation_image = np.uint8(lime_overlay_result[0] * 255)  # Ensure the image is in [0, 255] range for saving and displaying

        # Check the dtype of lime_explanation_image
        print(f"LIME Image dtype: {lime_explanation_image.dtype}")  # Should print uint8

        # Plot the LIME image
        axes[1, 1].imshow(lime_explanation_image)
        axes[1, 1].set_title('LIME Explanation')
        axes[1, 1].axis('off')
    else:
        print("LIME explanation not available.")

    # Add explanation text alongside LIME image
    axes[1, 2].axis('off')  # Turn off the axis for the text box
    explanation_text = """
    LIME Explanation:
    - The highlighted regions correspond to the important features
      that contributed to the model's prediction for the top class.
    - The model's prediction is based on these regions, indicating 
      areas of the brain tumor image that were critical for classification.
    """
    axes[1, 2].text(0, 0.5, explanation_text, ha='left', va='center', fontsize=10, wrap=True, color='black')

    # Adjust layout to ensure everything fits well with more padding
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Increase space between subplots
    plt.tight_layout(pad=4.0)  # Add more padding to ensure no overlap

    # Show the plot
    plt.show()

    # Save results including LIME explanation
    save_results(img, heatmap, superimposed, lime_explanation_image)

def save_results(original_img, heatmap, superimposed_img, lime_explanation_image, output_dir = 'output/'):
    # Save images
    original_img.save(f"{output_dir}original_image.png")
    plt.imsave(f"{output_dir}heatmap.png", heatmap, cmap='jet')
    superimposed_img.save(f"{output_dir}superimposed_image.png")
    
    # Save LIME image
    if lime_explanation_image is not None:
        lime_image_pil = Image.fromarray(lime_explanation_image)  # Convert LIME image to PIL format
        lime_image_pil.save(f"{output_dir}lime_explanation_image.png")
        print(f"LIME image saved as {output_dir}lime_explanation_image.png")

    print(f"Results saved in {output_dir}")


# Example usage
if __name__ == "__main__":
    # Apply Grad-CAM and LIME
    image_path = 'C:/Users/Medha Agarwal/Desktop/GANs/augmented_mri_images/augmented_mri_0.png'  # Update with the actual image path
    apply_gradcam_and_lime(image_path=image_path, model=model, opacity=0.5)
