from torchvision.models import googlenet, GoogLeNet_Weights
import torch.nn as nn

def construct_GoogLeNet_model(out_features):
    """
    Constructs a ResNet-18 model with a custom output layer.
    
    Parameters:
    out_features (int): Number of output classes.
    """
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, out_features)
    return model

if __name__ == "__main__":

    import torch

    model = construct_ResNet_model(6)
    print(model)

    # Test the model with a random input
    x = torch.randn(1, 3, 100, 100)  # Batch size of 1, 1 channel, 48x48 image
    output = model(x)
    print(output.shape)  # Should be (1, 16, 23, 23)