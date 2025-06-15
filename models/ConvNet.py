import torch
import torch.nn as nn

class ConvNet(nn.Module):
    """
    A Convolutional Network (ConvNet) model for image classification.
    This model consists of three convolutional layers followed by three fully connected layers.
    Each layer uses ReLU activation.
    Each convolutional layer is followed by a max pooling layer
    Each fully connected layer, except the last one, is followed by a dropout layer to prevent overfitting.
    The final output layer uses softmax to produce class probabilities.
    """
    def __init__(self, out_features):
        super(ConvNet, self).__init__()

        self.out_features = out_features

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3)

        self.fc1 = nn.Linear(32 * 10 * 10, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, out_features)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)

        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # (batch_size, 64, 49, 49)
        x = self.pool(self.relu(self.conv2(x))) # (batch_size, 64, 23, 23)
        x = self.pool(self.relu(self.conv3(x))) # (batch_size, 32, 10, 10)

        x = self.flatten(x)  # Flatten the tensor (batch_size, 128 * 12 * 12) = (batch_size, 18432)

        x = self.dropout(self.relu(self.fc1(x)))  # (batch_size, 512)
        x = self.dropout(self.relu(self.fc2(x))) # (batch_size, 64)
        x = self.fc3(x) # (batch_size, out_features)

        x = self.softmax(x)  # Apply softmax to the output

        return x
    

if __name__ == "__main__":
    model = ConvNet(6)
    print(model)

    # Test the model with a random input
    x = torch.randn(1, 3, 100, 100)  # Batch size of 1, 1 channel, 48x48 image
    output = model(x)
    print(output.shape)  # Should be (1, 16, 23, 23)