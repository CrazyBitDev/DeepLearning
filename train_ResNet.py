from models.ResNet import construct_ResNet_model

from train_function import train

def main():
    model = construct_ResNet_model(7)
    train(model, "ResNet18")



if __name__ == "__main__":
    main()