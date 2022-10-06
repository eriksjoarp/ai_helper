import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, fc_size=4096, fc_nums=3, debug=False):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=(4,4), padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(1,1), padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)
        self.fc1 = nn.Linear(in_features=9216, out_features=fc_size)
        self.fc2 = nn.Linear(in_features=4096, out_features=fc_size)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)
        self.fc_size = fc_size
        self.fc_nums = fc_nums

        if debug:
            print('fc_nums:' + str(fc_nums))
            print('fc_size:' + str(fc_size))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        if self.fc_nums==3:
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        else:
            x = self.fc2(x)
        return x


class Network(nn.Module):  # extend nn.Module class of nn
    def __init__(self):
        super().__init__()  # super class constructor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.batchN1 = nn.BatchNorm2d(num_features=6)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.batchN2 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):  # implements the forward method (flow of tensors)

        # hidden conv layer
        t = self.conv1(t)
        t = F.max_pool2d(input=t, kernel_size=2, stride=2)
        t = F.relu(t)
        t = self.batchN1(t)

        # hidden conv layer
        t = self.conv2(t)
        t = F.max_pool2d(input=t, kernel_size=2, stride=2)
        t = F.relu(t)

        # flatten
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.batchN2(t)
        t = self.fc2(t)
        t = F.relu(t)

        # output
        t = self.out(t)

        return t


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "MLP"


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, (5,5), (1,1))
        self.conv2 = nn.Conv2d(20, 50, (5,5), (1,1))
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"
