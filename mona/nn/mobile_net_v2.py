import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1), expansion=1, shortcut=True):
        super(Bottleneck, self).__init__()

        inner_channel = in_channels * expansion

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channel, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(inner_channel),
            # nn.ReLU6(),
            nn.Hardswish(),

            nn.Conv2d(inner_channel, inner_channel, kernel_size=(3, 3), stride=stride, groups=inner_channel, padding=1),
            nn.BatchNorm2d(inner_channel),
            # nn.ReLU6(),
            nn.Hardswish(),

            nn.Conv2d(inner_channel, out_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = shortcut

    def forward(self, x):
        h = self.conv(x)
        if self.shortcut:
            h += x
        return h


class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, expansion=6):
        super(MobileNetV2, self).__init__()

        # if in_height != 32:
        #     pass
        # if in_width % 32 != 0:
        #     pass

        # final_height = in_height // 32
        # final_width = in_width // 32

        # self.class_size = class_size

        self.conv1 = nn.Conv2d(in_channels, 32, stride=(2, 2), kernel_size=(3, 3), padding=1)
        self.bottlenecks = nn.Sequential(
            Bottleneck(32, 16, expansion=1, shortcut=False),

            Bottleneck(16, 24, stride=(2, 2), expansion=expansion, shortcut=False),
            Bottleneck(24, 24, expansion=expansion),

            Bottleneck(24, 32, stride=(2, 2), expansion=expansion, shortcut=False),
            Bottleneck(32, 32, expansion=expansion),
            Bottleneck(32, 32, expansion=expansion),

            Bottleneck(32, 64, stride=(2, 1), expansion=expansion, shortcut=False),
            Bottleneck(64, 64, expansion=expansion),
            Bottleneck(64, 64, expansion=expansion),
            Bottleneck(64, 64, expansion=expansion),

            Bottleneck(64, 96, expansion=expansion, shortcut=False),
            Bottleneck(96, 96, expansion=expansion),
            Bottleneck(96, 96, expansion=expansion),

            Bottleneck(96, 160, stride=(2, 1), expansion=expansion, shortcut=False),
            Bottleneck(160, 160, expansion=expansion),
            Bottleneck(160, 160, expansion=expansion),

            Bottleneck(160, 320, expansion=expansion, shortcut=False)
        )
        # -> N * 320 * width / 8 * height / 32

        self.conv2 = nn.Conv2d(320, 512, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv2(x)

        return x
