import torch
import torch.nn as nn
import torch.nn.functional as F


def get_nl(nl):
    if nl == "RE":
        return nn.ReLU()
    elif nl == "HS":
        return nn.Hardswish()


class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size=(3, 3),
                 squeeze_rate=4, nl="RE", has_se=False, stride=(1, 1)):
        super(MobileNetV3Block, self).__init__()

        self.has_se = has_se

        padding = 0
        if kernel_size[0] == 3:
            padding = 1
        elif kernel_size[0] == 5:
            padding = 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(exp_size),
            get_nl(nl),

            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, groups=exp_size, padding=padding),
            nn.BatchNorm2d(exp_size),
            get_nl(nl),
        )

        if has_se:
            self.se = SE(exp_size, squeeze_rate=squeeze_rate)

        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels and stride[0] == 1 and stride[1] == 1:
            self.shortcut = True
        else:
            self.shortcut = False

    def forward(self, x):
        x1 = x

        x = self.conv1(x)
        if self.has_se:
            x = self.se(x)

        x = self.conv2(x)

        if self.shortcut:
            x = x + x1

        return x


class SE(nn.Module):
    def __init__(self, channel, squeeze_rate=4):
        super(SE, self).__init__()

        self.linear1 = nn.Linear(channel, channel // squeeze_rate)
        self.linear2 = nn.Linear(channel // squeeze_rate, channel)

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.pool(x)
        y = torch.squeeze(y, 3)
        y = torch.squeeze(y, 2)
        y = self.linear1(y)
        y = F.relu(y)
        y = self.linear2(y)
        y = F.hardsigmoid(y)
        y = torch.unsqueeze(y, 2)
        y = torch.unsqueeze(y, 3)
        x = x * y
        return x


class MobileNetV3Small(nn.Module):
    def __init__(self, out_size, in_channels=3):
        super(MobileNetV3Small, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=1), # -> H // 2, W // 2
            nn.BatchNorm2d(16),
            nn.Hardswish(),

            MobileNetV3Block(16, 16, exp_size=16, has_se=True, nl="RE", kernel_size=(3, 3), stride=(2, 2)),
            MobileNetV3Block(16, 24, exp_size=64, has_se=False, nl="RE", kernel_size=(3, 3), stride=(2, 2)),           # original (2, 2)
            MobileNetV3Block(24, 24, exp_size=88, has_se=False, nl="RE", kernel_size=(3, 3), stride=(1, 1)),
            MobileNetV3Block(24, 40, exp_size=96, has_se=True, nl="HS", kernel_size=(5, 5), stride=(2, 2)),            # original (2, 2)
            MobileNetV3Block(40, 40, exp_size=240, has_se=True, nl="HS", kernel_size=(5, 5), stride=(1, 1)),
            MobileNetV3Block(40, 40, exp_size=240, has_se=True, nl="HS", kernel_size=(5, 5), stride=(1, 1)),
            MobileNetV3Block(40, 48, exp_size=120, has_se=True, nl="HS", kernel_size=(5, 5), stride=(1, 1)),
            MobileNetV3Block(48, 48, exp_size=144, has_se=True, nl="HS", kernel_size=(5, 5), stride=(1, 1)),
            MobileNetV3Block(48, 96, exp_size=288, has_se=True, nl="HS", kernel_size=(5, 5), stride=(2, 1)),           # original (2, 2)
            MobileNetV3Block(96, 96, exp_size=576, has_se=True, nl="HS", kernel_size=(5, 5), stride=(1, 1)),
            MobileNetV3Block(96, 96, exp_size=576, has_se=True, nl="HS", kernel_size=(5, 5), stride=(1, 1)),   # N, 96, H // 32, W // 32

            nn.Conv2d(96, out_size, kernel_size=(1, 1), stride=(1, 1)),  # N, out_size, H // 32, W // 32
            SE(out_size),
        )

    def forward(self, x):
        return self.conv(x)