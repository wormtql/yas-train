import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), padding=1):
        super(Block, self).__init__()

        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            padding=padding,
            groups=in_channels,
            stride=stride,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.pw_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pw_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


class MobileNetV1(nn.Module):
    def __init__(self, in_channels):
        super(MobileNetV1, self).__init__()

        # N, 3, H, W
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_2_to_13 = nn.Sequential(
            Block(32, 64),
            Block(64, 128, stride=(2, 2)),
            Block(128, 128),
            Block(128, 256, stride=(2, 2)),
            Block(256, 256),
            Block(256, 512, stride=(2, 1)),
            Block(512, 512),
            Block(512, 512),
            Block(512, 512),
            Block(512, 512),
            Block(512, 512),
            Block(512, 512, stride=(2, 1)),
        )

        # self.pool = nn.AvgPool2d(7)
        # self.linear = nn.Linear(1024, 1000)

    # N * 3 * W * 32
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_2_to_13(x)
        # x = self.pool(x)

        return x