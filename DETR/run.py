from transformers import AutoImageProcessor
from transformers import DetrForObjectDetection
from detr import DetrModel
from PIL import Image
import requests
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrModel.from_pretrained("facebook/detr-resnet-50")
# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")
# forward pass
outputs = model(**inputs)
# the last hidden states are the final query embeddings of the Transformer decoder
# these are of shape (batch_size, num_queries, hidden_size)
last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)
[1, 100, 256]