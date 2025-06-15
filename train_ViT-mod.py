from models.ViT import construct_ViT_model

from train_function import train

def main():
    model = construct_ViT_model(7, image_size=100, patch_size=10)
    train(
        model, "ViT-modTest",
        image_size=(100, 100), lr=5e-4,
        scheduler=True
    )



if __name__ == "__main__":
    main()