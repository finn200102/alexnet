import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96,
                               kernel_size=11, stride=4, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2,
                                    padding=0)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.maxpool2 = nn.MaxPool2d(3, 2, 0)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(3, 2, 0)
        self.fc1 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        print("After MaxPool1:", x.shape)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        print("After MaxPool2:", x.shape)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        print("After MaxPool3:", x.shape)

        x = x.flatten(1)
        print("After Flatten:", x.shape)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x




def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Mps is availabe")
    else:
        device = torch.cpu("cpu")

    model = AlexNet().to(device)
    x = torch.randn(1, 3, 224, 224).to(device)
    y = model.forward(x)
    print(y.shape)

if __name__ == "__main__":
    main()
