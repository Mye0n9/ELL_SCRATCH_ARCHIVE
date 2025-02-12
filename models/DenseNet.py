import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicLayer(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p_dropout = p_dropout
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1, bias = False)
    
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = F.dropout(x, p=self.p_dropout, training = self.training)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compact, l, m_block, c_block):
        super().__init__()
        self.in_channels = in_channels
        self.compact = compact
        self.l = l
        self.m_block = m_block
        self.c_block = c_block

        self.internalLayer = nn.Sequential(
            nn.Conv2d(self.in_channels, int(math.floor(self.in_channels*self.compact)), kernel_size = 1, padding=0, stride = 1),
            nn.AvgPool2d(2)
        )
        self.lastLayer = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        if self.m_block == self.c_block:
            x = self.lastLayer(x)
        else: x = self.internalLayer(x)
        # print(x.shape)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, k, l, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.k = k # growth rate
        self.l = l # number of layers
        self.p_dropout = p_dropout

        self.layers = nn.ModuleList()
        for _ in range(self.l-1):
            self.layers.append(BasicLayer(self.in_channels, self.k, self.p_dropout))
            self.in_channels += self.k
        self.in_channels -= self.k

    def forward(self, x):
        for layer in self.layers:
            origin_x = x
            x = layer(x)
            x = torch.cat((origin_x, x),dim=1)
        return x

class DenseNet(nn.Module):
    def __init__(self, n_classes, n_blocks, in_channels, k, l, p_dropout, compact):
        super().__init__()
        self.n_classes = n_classes
        self.n_blocks = n_blocks

        self.in_channels = in_channels
        self.k = k
        self.l = l
        self.p_dropout = p_dropout
        self.compact = compact

        self.conv = nn.Conv2d(in_channels = 3, out_channels = self.in_channels, kernel_size=3, padding=1, bias=False, stride = 1)
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            self.blocks.append(DenseBlock(in_channels=self.in_channels, k=self.k, l=self.l,p_dropout=self.p_dropout))
            self.in_channels += self.k*(self.l-1)
            self.blocks.append(TransitionLayer(self.in_channels,self.compact,self.l,self.n_blocks, i+1))
            self.in_channels = int(math.floor(self.in_channels*self.compact))
        
        self.fc = nn.Linear(int(self.in_channels/self.compact), self.n_classes)
    
    def forward(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x 

# x = torch.randn(10,3,32,32)
# model = DenseNet(10, 3, 16, 12, 5, 0.15, 0.5)
# pred = model(x)
# print(pred.shape)