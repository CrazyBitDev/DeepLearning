
from models.VGG import construct_VGG_model

from train_function import train

def main():
    model = construct_VGG_model(7)
    train(model, "VGG16")



if __name__ == "__main__":
    main()