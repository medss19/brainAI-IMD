from tensorflow.keras.models import load_model
# Load your trained model
model = load_model('best_mri_model.h5')

# Print the model summary
model.summary()
