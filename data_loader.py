# data_loader.py

from torchvision import transforms
from PIL import Image

# Define the standard preprocessing transform
transform = transforms.Compose([
    transforms.Resize((256, 128)),  # Resize images to ReID standard size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(           # Normalize based on ImageNet stats
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def load_and_preprocess(image_path):
    """
    Loads an image from the given path and applies standard preprocessing.
    Returns a PyTorch tensor.
    """
    image = Image.open(image_path).convert("RGB")
    return transform(image)
