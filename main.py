import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2(in_channels=3, out_channels=96,
                              kernel_size=11, stride=4, padding=0)
        self.maxpool1 = nn.MaxPool2(kernel_size=3, stride=2,
                                    padding=0)
        self.conv2 = nn.Conv2(96, 256, 5, 1, 2)
        self.maxpool2 = nn.MaxPool2(3, 2, 0)
        self.conv3 = nn.Conv2(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2(384, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2(3, 2, 0)
        self.fc1 = nn.Linear(in_features=6*6*256, out_features=4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Mps is availabe")
    else:
        device = torch.cpu("cpu")

if __name__ == "__main__":
    main()
