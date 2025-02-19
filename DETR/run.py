from transformers import AutoImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import requests

# Load the test image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Load the image processor and model
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Prepare the image for the model
inputs = image_processor(images=image, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Convert raw outputs to meaningful object detection results
target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# Display the results
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"Detected {model.config.id2label[label.item()]} with confidence {score.item():.3f} at {box.tolist()}")
