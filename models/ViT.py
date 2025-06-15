from torchvision.models import vit_b_16, VisionTransformer, ViT_B_16_Weights
import torch.nn as nn

def construct_ViT_model(out_features, image_size=None, patch_size=None):
    """
    Constructs a Vision Transformer (ViT) model

    Parameters:
    out_features (int): Number of output classes
    image_size (int, optional): Size of the input image. If None, uses the default size for ViT.
    patch_size (int, optional): Size of the patches. If None, uses the default size for ViT.
    """

    # If image_size and patch_size are provided, create a custom VisionTransformer
    if image_size is not None and patch_size is not None:
        model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=out_features,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072
        )
    else:
        # Use the pre-trained ViT-B/16 model and modify the output layer to match out_features
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, out_features)
    return model

if __name__ == "__main__":

    import torch

    model = construct_ViT_model(6)
    print(model)

    # Test the model with a random input
    x = torch.randn(3, 3, 224, 224)  # Batch size of 3, 3 channels, 224x224 image
    output = model(x)
    print(output.shape)  # Should be (3, 6) for 6 classes