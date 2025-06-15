import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

def construct_VGG_model(out_features):
    model = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, out_features)
    return model

if __name__ == "__main__":

    import torch

    model = construct_VGG_model(6)
    print(model)

    # Test the model with a random input
    x = torch.randn(16, 3, 224, 224) 
    output = model(x)
    print(output.shape)