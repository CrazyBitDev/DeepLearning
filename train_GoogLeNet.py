from models.GoogLeNet import construct_GoogLeNet_model

from train_function import train

def main():
    model = construct_GoogLeNet_model(7)
    train(model, "GoogLeNet")



if __name__ == "__main__":
    main()