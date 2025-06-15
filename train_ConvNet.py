from models.ConvNet import ConvNet

from train_function import train

def main():
    model = ConvNet(7)
    train(
        model, "ConvNet",
        image_size = (100, 100),
    )



if __name__ == "__main__":
    main()