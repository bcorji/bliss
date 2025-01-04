# Clone the YOLOv5 repository and install dependencies
#!git clone https://github.com/ultralytics/yolov5
#%cd yolov5
#!pip install -r requirements.txt

# Data Objects

from pathlib import Path
import torch

# Load the YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Path to the image
image_path = 'path_to_image.jpg'  # Replace with your image path

# Perform object detection
results = model(image_path)

# Display results
results.show()  # Show image with bounding boxes
results.save(Path('./output'))  # Save results to the 'output' directory

# Print detected objects
print(results.pandas().xyxy[0])  # Print predictions as a DataFrame