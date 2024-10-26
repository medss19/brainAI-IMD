import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Path to your original dataset organized by class
original_dataset_dir = 'brain-tumor-mri-dataset/Training'

# Image size and batch size
image_size = (128, 128)
batch_size = 32

# Data generator for the original dataset
datagen = ImageDataGenerator(rescale=1./255)

# Load the original dataset
original_generator = datagen.flow_from_directory(
    original_dataset_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Ensure image order is consistent
)

# Get the class labels from the original generator
class_indices = original_generator.class_indices
print(f"Class indices: {class_indices}")

# Save original images and labels in arrays
original_images, original_labels = [], []

for batch_images, batch_labels in original_generator:
    original_images.append(batch_images)
    original_labels.append(batch_labels)
    if len(original_images) * batch_size >= original_generator.samples:
        break

# Convert to numpy arrays
original_images = np.vstack(original_images)
original_labels = np.vstack(original_labels)

print(f"Original dataset shape: {original_images.shape}, Labels shape: {original_labels.shape}")

# Load augmented images
augmented_dir = 'augmented_mri_images'
num_augmented = 5000  # Example number of augmented images

# Load augmented images (assuming grayscale)
augmented_images = []
for i in range(num_augmented):
    img_path = os.path.join(augmented_dir, f'augmented_mri_{i}.png')
    img = load_img(img_path, target_size=image_size, color_mode='grayscale')
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    augmented_images.append(img_array)

augmented_images = np.array(augmented_images)
print(f"Augmented dataset shape: {augmented_images.shape}")

# Reshape to add the channel dimension if needed
if len(augmented_images.shape) == 3:
    augmented_images = augmented_images.reshape(augmented_images.shape[0], 128, 128, 1)

# Convert augmented grayscale images to RGB (1 channel to 3 channels)
augmented_images_rgb = np.repeat(augmented_images, 3, axis=-1)
print(f"Converted augmented images to RGB: {augmented_images_rgb.shape}")

# Calculate the number of samples per class in the original dataset
samples_per_class = original_labels.sum(axis=0)

# Calculate the proportion of each class
class_proportions = samples_per_class / samples_per_class.sum()

# Calculate how many augmented samples should be in each class
augmented_samples_per_class = (class_proportions * num_augmented).astype(int)

# Adjust the last class to ensure the total is exactly num_augmented
augmented_samples_per_class[-1] = num_augmented - augmented_samples_per_class[:-1].sum()

# Create augmented labels
augmented_labels = np.zeros((num_augmented, len(class_indices)))
start_idx = 0
for class_idx, num_samples in enumerate(augmented_samples_per_class):
    end_idx = start_idx + num_samples
    augmented_labels[start_idx:end_idx, class_idx] = 1
    start_idx = end_idx

print(f"Augmented labels shape: {augmented_labels.shape}")

# Combine original and augmented datasets
combined_images = np.vstack((original_images, augmented_images_rgb))
combined_labels = np.vstack((original_labels, augmented_labels))

print(f"Combined dataset shape: {combined_images.shape}, Labels shape: {combined_labels.shape}")

# Shuffle the combined dataset
shuffle_indices = np.arange(combined_images.shape[0])
np.random.shuffle(shuffle_indices)
combined_images = combined_images[shuffle_indices]
combined_labels = combined_labels[shuffle_indices]

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(combined_images, combined_labels, test_size=0.2, random_state=42)

# Load ResNet50 without the top layers
input_shape = (128, 128, 3)  # RGB images for ResNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

# Freezing the base model layers
base_model.trainable = False

# Add custom layers on top of ResNet
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # Output for 4 classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set up model checkpointing
checkpoint = ModelCheckpoint('best_mri_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=batch_size, callbacks=[checkpoint])

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(x_val, y_val)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Optional: Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()