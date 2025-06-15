from models.ViT import construct_ViT_model

from train_function import train

def main():
    model = construct_ViT_model(7)
    train(
        model, "ViT-B16Test",
        lr=1e-5, scheduler=True
    )



if __name__ == "__main__":
    main()