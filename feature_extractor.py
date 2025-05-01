# feature_extractor.py

import torch
from torchvision import models  
import torch.nn as nn

def get_model(device='cpu'):
    """
    Loads a pretrained ResNet50 model and removes its classification layer.
    Returns the modified model in evaluation mode.
    """
    model = models.resnet50(pretrained=True)  # Load model trained on ImageNet
    model.fc = nn.Identity()  # Remove classification layer, output is 2048-D features
    model = model.to(device)
    model.eval()  # Set to inference mode
    return model

def extract_features(model, image_tensor, device='cpu'):
    """
    Passes an image tensor through the model to get a feature embedding.
    Returns a numpy array.
    """
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():  # Disable gradient computation for faster inference
        features = model(image_tensor).cpu().numpy()
    return features
