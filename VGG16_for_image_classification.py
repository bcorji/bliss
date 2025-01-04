import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet')

# Load and preprocess an image
image_path = 'path_to_image.jpg'  # Replace with your image path
img = load_img(image_path, target_size=(224, 224))  # Resize image to 224x224
img_array = img_to_array(img)  # Convert to numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = preprocess_input(img_array)  # Preprocess the image

# Predict the class
predictions = model.predict(img_array)
decoded_predictions = decode_predictions(predictions, top=3)  # Get top 3 predictions

# Display the predictions
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label} ({score:.2f})")