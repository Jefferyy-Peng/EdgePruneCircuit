from transformers import AutoImageProcessor

from src.utils import ImageNetDataset

imagenet_val = ImageNetDataset(root_dir='/data/nvme1/yxpeng/imagenet/val',
                                   processor=AutoImageProcessor.from_pretrained("google/vit-base-patch16-224"))

# Create DataLoader
batch_size = 16  # Adjust as needed
val_loader = DataLoader(imagenet_val, batch_size=batch_size, shuffle=True)

vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
vit_model.eval()