import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # F(x)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False), 
            nn.BatchNorm2d(out_channels)
        )

        self.down_sample = None
        if self.in_channels != self.out_channels or stride != 1:
            # downsampling
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.down_sample:
            identity = self.down_sample(x) # perform down sampling if it is needed
        
        res = self.conv_block(x)
        res = F.relu(identity + res)  # identity shortcut
        return res


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_channels = 64
        
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 4 blocks for ResNet34
        self._one = self.stack_layer(out_channels=64, num_blocks=3, check_first_layer=True)
        self._two = self.stack_layer(out_channels=128, num_blocks=4)
        self._three = self.stack_layer(out_channels=256, num_blocks=6)
        self._four = self.stack_layer(out_channels=512, num_blocks=3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def stack_layer(self, out_channels, num_blocks, check_first_layer=False):
        layers = []
        for i in range(num_blocks):
            stride = 1 if i != 0 or check_first_layer else 2
            block = ResBlock(self.in_channels, out_channels, stride=stride)
            layers.append(block)
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base(x) # base
        x = self._one(x) 
        x = self._two(x)
        x = self._three(x)
        x = self._four(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
