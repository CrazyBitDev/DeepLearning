from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def construct_ResNet_model(out_features):
    """
    Constructs a ResNet-18 model with a custom output layer.
    
    Parameters:
    out_features (int): Number of output classes.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model