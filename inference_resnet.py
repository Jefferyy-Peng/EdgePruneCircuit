import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import requests

# Step 1: Load a pretrained ResNet-50 model
model = resnet50(pretrained=True)
model.eval()

# Step 2: Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),             # Resize image to 224x224
    transforms.ToTensor(),                     # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# Step 3: Load an example image from ImageNet
# Example image URL (Labrador Retriever)
image_url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
response = requests.get(image_url, stream=True)
image = Image.open(response.raw).convert("RGB")  # Ensure it's RGB

# Step 4: Preprocess the image
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Step 5: Perform inference
with torch.no_grad():
    output = model(input_tensor)

# Step 6: Get top-5 predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# Step 7: Load ImageNet class labels
import json
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = json.loads(requests.get(LABELS_URL).text)

# Step 8: Print predictions
for i in range(5):
    print(f"Class: {labels[top5_catid[i]]}, Probability: {top5_prob[i].item():.4f}")
