# from tensorflow.keras.models import load_model
# # Load your trained model
# model = load_model('brain_tumor_classifier.h5')

# # Print the model summary
# model.summary()

from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('brain_tumor_classifier.h5')

# Print only the names of each layer
for layer in model.layers:
    print(layer.name)
