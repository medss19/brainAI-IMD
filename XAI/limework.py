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
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img
# from your_project import preprocess_input, get_gradcam_heatmap  # Ensure these are correctly imported


# Load the ResNet50 model (pre-trained weights)
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Grad-CAM function to generate the heatmap
def get_gradcam_heatmap(model, image, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.inputs], [model.get_layer(
        last_conv_layer_name).output, model.output])

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


# LIME prediction function
def predict_fn(images):
    images = np.array(images)
    images = preprocess_input(images)
    return model.predict(images)


# Function to visualize Grad-CAM and LIME with an explanation
def apply_gradcam_and_lime(image_path, model, target_size=(128, 128), opacity=0.4):
    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_array_preprocessed = preprocess_input(img_array_expanded)

    # Specify the last conv layer for ResNet50
    last_conv_layer_name = 'conv5_block3_out'
    heatmap, predictions = get_gradcam_heatmap(model, img_array_preprocessed, last_conv_layer_name)

    # Prediction probabilities
    class_names = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary']
    confidences = {class_names[i]: predictions[0][i] for i in range(len(class_names))}

    # LIME Explanation
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array.astype('double'), predict_fn, top_labels=4, hide_color=0, num_samples=1000)

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=80)

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Heatmap
    axes[0, 1].imshow(heatmap, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap')
    axes[0, 1].axis('off')

    # Superimposed Image
    img_array = img_to_array(img)
    heatmap_img = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_img]

    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(target_size)
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed = jet_heatmap * opacity + img_array
    superimposed = array_to_img(superimposed)

    axes[0, 2].imshow(superimposed)
    axes[0, 2].set_title('Superimposed Image')
    axes[0, 2].axis('off')

    # Bar chart for confidence scores
    axes[1, 0].barh(list(confidences.keys()), list(confidences.values()), color='skyblue')
    axes[1, 0].set_title('Prediction Confidence')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_xlim(0, 1)

    # LIME Explanation Overlay
    lime_overlay_result = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=True)
    
    if lime_overlay_result:
        lime_explanation_image = np.uint8(lime_overlay_result[0] * 255)

        # Plot the LIME image
        axes[1, 1].imshow(lime_explanation_image)
        axes[1, 1].set_title('LIME Explanation')
        axes[1, 1].axis('off')
    else:
        print("LIME explanation not available.")

    # Explanation text
    axes[1, 2].axis('off')
    explanation_text = """
    LIME Explanation:
    - The highlighted regions correspond to the important features
      that contributed to the model's prediction for the top class.
    - The model's prediction is based on these regions, indicating 
      areas of the brain tumor image that were critical for classification.
    """
    axes[1, 2].text(0, 0.5, explanation_text, ha='left', va='center', fontsize=10, wrap=True, color='black')

    # Adjust layout and show the plot
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout(pad=4.0)

    # Save results including LIME explanation
    output_dir = 'static/output/'
    os.makedirs(output_dir, exist_ok=True)
    
    original_image_path = os.path.join(output_dir, 'original_image.png')
    heatmap_image_path = os.path.join(output_dir, 'heatmap.png')
    superimposed_image_path = os.path.join(output_dir, 'superimposed_image.png')
    lime_image_path = os.path.join(output_dir, 'lime_explanation_image.png')

     # Save the images using PIL for better quality
    img.save(original_image_path, format='PNG', dpi=(300, 300))  # Save with high DPI
    Image.fromarray(np.uint8(255 * heatmap)).save(heatmap_image_path, format='PNG', dpi=(300, 300))
    superimposed.save(superimposed_image_path, format='PNG', dpi=(300, 300))
    Image.fromarray(lime_explanation_image).save(lime_image_path, format='PNG', dpi=(300, 300))


    plt.close(fig)

    return {
        'original': original_image_path,
        'heatmap': heatmap_image_path,
        'superimposed': superimposed_image_path,
        'lime': lime_image_path
    }


def save_results(original_img, heatmap, superimposed_img, lime_explanation_image, output_dir='static/output/'):
    os.makedirs(output_dir, exist_ok=True)

    # Save images
    original_img.save(os.path.join(output_dir, "original_image.png"))
    plt.imsave(os.path.join(output_dir, "heatmap.png"), heatmap, cmap='jet')
    superimposed_img.save(os.path.join(output_dir, "superimposed_image.png"))

    if lime_explanation_image is not None:
        lime_image_pil = Image.fromarray(lime_explanation_image)
        lime_image_pil.save(os.path.join(output_dir, "lime_explanation_image.png"))

    # Return the path of the saved result images to be used in Flask
    return 'output/'
