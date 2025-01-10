import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests

# Load an image from a URL
url = "https://example.com/image.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Load pre-trained ViT feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
inputs = feature_extractor(images=image, return_tensors="pt")

# Load the ViT model for image classification
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Perform inference
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

# Print the predicted class index
print(f"Predicted class index: {predicted_class_idx}")