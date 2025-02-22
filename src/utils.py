import torch
from torchvision import datasets
from PIL import Image

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, processor):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
        inputs = self.processor(images=image, return_tensors="pt")  # Use processor
        return inputs["pixel_values"].squeeze(0), label