import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,dropout_rate):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.block(x)

drop_type = ''
n_path = -1

class JoinLayer(nn.Module):
    def __init__ (self, n_columns):
        super(JoinLayer,self).__init__()
        self.n_columns = n_columns
    
    def forward(self, paths):
        if len(paths) == 1:
            # C = 2
            res = paths[0]
        else:
            # local and global drop
            if drop_type == 'local':
                res = [path for path in paths if torch.rand(1).item() > 0.15]
                if len(res) == 0:
                    res.append(random.choice(paths))

            elif drop_type == 'global':
                kept_path_index = max(0, len(paths) - n_path)
                res [paths[kept_path_index]] if len(paths) >= n_path else [torch.zeros_like(paths[0])]
            
            else: res = paths
            
            res = torch.mean(torch.stack(res), dim=0)        

        return res


class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_columns, p_dropout):
        super(FractalBlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p_dropout = p_dropout

        self.n_columns = n_columns

        self.conv_path = ConvBlock(self.in_channels, self.out_channels, self.p_dropout)

        if self.n_columns > 1:
            self.fractal_path = nn.ModuleList([
                FractalBlock(in_channels=self.in_channels, out_channels= self.out_channels, n_columns=self.n_columns - 1, p_dropout= self.p_dropout),
                JoinLayer(n_columns-1),
                FractalBlock(in_channels=self.out_channels, out_channels= self.out_channels, n_columns=self.n_columns - 1, p_dropout= self.p_dropout)
            ])
        else:
            self.fractal_path = None
    
    def forward(self,x):
        paths = []
        paths.append(self.conv_path(x)) # extend로 넣게 되면 [batch_size, ch, w, h]가 아니라 batch_size * [ch, w, h]로 넣음
        if self.fractal_path is not None:
            for path in self.fractal_path:
                x = path(x)

            paths.extend(x)
        return paths

class FractalNet(nn.Module):
    global drop_type, n_path
    def __init__(self, n_blocks, n_columns, n_classes):
        super().__init__()
        self.n_columns = n_columns
        self.n_blocks = n_blocks

        self.conv = ConvBlock(3,64,0)
        self.fractalBlock = nn.ModuleList()

        in_channel = 64
        out_channel = 128

        for _ in range(self.n_blocks):
            self.fractalBlock.append(FractalBlock(in_channels=in_channel, out_channels=out_channel,n_columns=n_columns, p_dropout=0.1))
            in_channel = out_channel
            out_channel *= 2

        self.transition = nn.MaxPool2d(kernel_size=2)
        self.final_transition = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(out_channel//2, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # drop_type 정하고, global이면 n_path도 정하기
        drop_type = "local" if random.random() <= 0.5 else "global"
        if drop_type == 'global':
            n_path = random.randint(1, self.n_columns)

        res = self.conv(x)
        for n, block in enumerate(self.fractalBlock):
            res = block(res)
            res = transition(res)
            if n == self.n_blocks-1:
                res = self.final_transition(res)
            else:
                res = self.transition(res)
        res = res.flatten(start_dim = 1)
        res = self.fc(res)
        return res
    
def transition(inputs):
    out = torch.stack(inputs,dim=0)
    out = torch.mean(out, dim=0)
    return out

# Test
# x = torch.randn(100, 3, 32, 32)
# model = FractalNet(n_blocks=3,n_columns=4,n_classes=10)

# res = model(x)
# print(res.shape)
