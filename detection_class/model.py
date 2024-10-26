from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Set path to the combined dataset
combined_data_dir = 'path_to_combined_dataset'

# Data augmentation and rescaling
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Training generator
train_generator = datagen.flow_from_directory(
    combined_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training')  # Use 80% of data for training

# Validation generator
validation_generator = datagen.flow_from_directory(
    combined_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # Use 20% of data for validation


# Load the VGG16 model pre-trained on ImageNet and exclude the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom classification layers
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, notumor, pituitary
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

model.save('brain_tumor_classifier.h5')
