import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ResidualBlock(nn.Module):
    # 3 groups of 18 residual blocks
    # 처음 들어갈때, downsampling
    def __init__(self, in_channels,out_channels,stride,p_survival):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.p_survival = p_survival

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
        # Survival probability 적용
        # training이랑 아닐때, 다르게 적용해보자
        if self.training:
            if random.random() < self.p_survival:
                out = F.relu(res + identity)  # identity shortcut
            else:
                out = identity
        else:
            out = F.relu(self.p_survival*res + identity)
        return out

class StochasticDepth(nn.Module):
    def __init__(self, n_classes, n_groups, survival_type , p_L):
        super().__init__()
        
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.survival_type = survival_type
        self.p_L = p_L
        
        self.in_channels = 16

        self.p_zero = 1

        self.base = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU()
        )

        # 3 groups for StochasticDepth
        self.groups = nn.ModuleList()
        first_group = True

        for i in range(self.n_groups):
            if self.survival_type == 'linear_decay':
                p_survival = 1-(i/self.n_groups)*(1-self.p_L)
                self.groups.append(self.stack_group(out_channels=self.in_channels*2, num_blocks=18, p_l = p_survival, check_first_layer = first_group))
            else:
                self.groups.append(self.stack_group(out_channels=self.in_channels*2, num_blocks=18, p_l = self.p_L, check_first_layer = first_group))
            first_group = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.in_channels, self.n_classes)

    def stack_group(self, out_channels, num_blocks, p_l, check_first_layer=False):
        layers = []
        for i in range(num_blocks):
            stride = 1 if i != 0 or check_first_layer else 2
            block = ResidualBlock(self.in_channels, out_channels, stride=stride, p_survival=p_l)
            layers.append(block)
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.base(x) # base
        for group in self.groups:
            x = group(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

# x = torch.randn(10,3,32,32)
# model = StochasticDepth(n_classes = 10, n_groups = 3, survival_type= 'uniform',p_L=0.5)
# pred = model(x)
# print(pred.shape)

