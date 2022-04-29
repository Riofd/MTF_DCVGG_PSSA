import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch


class DualVGGNet(nn.Module):
    def __init__(self, arch, num_classes=3):
        super(DualVGGNet, self).__init__()
        self.in_channels = 3
        self.conv3_16_rotor = self.__make_layer(16, arch[0])
        self.conv3_32_rotor = self.__make_layer(32, arch[1])
        self.conv3_64_rotor = self.__make_layer(64, arch[2])
        self.conv3_128a_rotor = self.__make_layer(128, arch[3])
        self.conv3_128b_rotor = self.__make_layer(128, arch[4])
        self.in_channels = 3
        self.conv3_16_vol = self.__make_layer(16, arch[0])
        self.conv3_32_vol = self.__make_layer(32, arch[1])
        self.conv3_64_vol = self.__make_layer(64, arch[2])
        self.conv3_128a_vol = self.__make_layer(128, arch[3])
        self.conv3_128b_vol = self.__make_layer(128, arch[4])

        self.fc1 = nn.Linear(2 * 7 * 7 * 128, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(4096)
        # self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(1024, num_classes)

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        out1 = F.max_pool2d(self.conv3_16_rotor(x1), 2)
        out1 = F.max_pool2d(self.conv3_32_rotor(out1), 2)
        out1 = F.max_pool2d(self.conv3_64_rotor(out1), 2)
        out1 = F.max_pool2d(self.conv3_128a_rotor(out1), 2)
        out1 = F.max_pool2d(self.conv3_128b_rotor(out1), 2)
        out1 = out1.view(out1.size(0), -1)

        out2 = F.max_pool2d(self.conv3_16_vol(x2), 2)
        out2 = F.max_pool2d(self.conv3_32_vol(out2), 2)
        out2 = F.max_pool2d(self.conv3_64_vol(out2), 2)
        out2 = F.max_pool2d(self.conv3_128a_vol(out2), 2)
        out2 = F.max_pool2d(self.conv3_128b_vol(out2), 2)
        out2 = out2.view(out2.size(0), -1)

        out = torch.cat([out1, out2], dim=1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.dropout(out, p=0.5, training=self.training)
        # out = F.relu(out)
        # out = self.fc2(out)
        # out = self.bn2(out)
        out = F.relu(out)
        return F.softmax(self.fc3(out))


def VGG_11():
    return DualVGGNet([1, 1, 2, 2, 2], num_classes=3)


def VGG_13():
    return DualVGGNet([1, 1, 2, 2, 2], num_classes=3)


def VGG_16():
    return DualVGGNet([2, 2, 3, 3, 3], num_classes=3)


def VGG_19():
    return DualVGGNet([2, 2, 4, 4, 4], num_classes=3)


def VGG_8():
    return DualVGGNet([1, 1, 1, 1, 1], num_classes=3)

