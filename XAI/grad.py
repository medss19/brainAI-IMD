import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

def get_layer_and_model(model):
    """
    Get the last convolutional layer and create appropriate model for Grad-CAM
    """
    # Get the ResNet50 base model
    resnet_model = model.get_layer('resnet50')
    
    # Find the last convolutional layer
    last_conv_layer = None
    for layer in resnet_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
    
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model")
    
    # Create a model that goes from the input to both the final layer and the last conv layer
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )
    
    return grad_model, last_conv_layer

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Create a Grad-CAM heatmap
    """
    grad_model, last_conv_layer = get_layer_and_model(model)

    # Compute the gradient of the top predicted class
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Extract gradients
    grads = tape.gradient(class_channel, conv_output)
    
    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by importance
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.maximum(tf.reduce_max(heatmap), 1e-10)
    
    return heatmap.numpy()

def apply_gradcam(image_path, model, target_size=(128, 128)):
    """
    Apply Grad-CAM to an image and display the result
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Generate the Grad-CAM heatmap
    try:
        heatmap = make_gradcam_heatmap(img_array, model)
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        print("Model summary:")
        model.summary()
        return

    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    # Superimposed
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    heatmap_img = np.uint8(255 * heatmap)
    
    # Use jet colormap
    jet = plt.cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_img]
    
    # Resize and convert
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize(target_size)
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose
    superimposed = jet_heatmap * 0.4 + img_array
    superimposed = tf.keras.preprocessing.image.array_to_img(superimposed)

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed)
    plt.title('Superimposed')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load model
    model = tf.keras.models.load_model('best_mri_model.h5')
    
    # Could you print the model summary to help debug?
    print("Model layers:")
    for layer in model.layers:
        print(f"Layer name: {layer.name}, Type: {type(layer)}")
    
    # Apply Grad-CAM
    image_path = 'C:/Users/Medha Agarwal/Desktop/GANs/augmented_mri_images/augmented_mri_0.png'
    apply_gradcam(
        image_path=image_path,
        model=model
    )