import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# Load an image from a URL
url = "https://example.com/image.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Load pre-trained CLIP processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Preprocess the image and text
inputs = processor(text=["a photo of a dog", "a photo of a cat"], images=image, return_tensors="pt", padding=True)

# Perform inference
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# Print the probabilities
print(probs)