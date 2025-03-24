import torch
import torch.nn as nn
import torch.nn.functional as F


class AFSUnit(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(AFSUnit, self).__init__()
        self.conv7 = nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3, groups=mid_channels)
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, groups=mid_channels)

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x1 = F.relu(self.conv7(x))
        x1 = self.alpha * x1 + x
        
        x2 = F.relu(self.conv1(x1))
        x2 = self.beta * x2 + x1
        
        x3 = F.relu(self.conv3(x2))
        out = self.gamma * x3 + x2

        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class ODSModule(nn.Module):
    def __init__(self, in_channels):
        super(ODSModule, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(in_channels * 8, in_channels * 4, kernel_size=1)
        
        self.afs1 = AFSUnit(in_channels * 4, in_channels * 4)
        self.afs2 = AFSUnit(in_channels * 4, in_channels * 4)
        self.afs3 = AFSUnit(in_channels * 4, in_channels * 4)
        self.ca_layer = CALayer(in_channels * 4)
        
        self.conv_final = nn.Conv2d(in_channels * 4, in_channels * 2, kernel_size=1)
        
        self.sigma = nn.Parameter(torch.ones(1))

    def forward(self, F):
        assert F.dim() == 4, "Shape error"
        xl = self.downsample(F)  # Resulting size: B x C x H/2 x W/2
        
        xa = F[:, :, 0::2, 0::2]  # Top-left sampling
        xb = F[:, :, 0::2, 1::2]  # Top-right sampling
        xc = F[:, :, 1::2, 0::2]  # Bottom-left sampling
        xd = F[:, :, 1::2, 1::2]  # Bottom-right sampling

        xh = [xa - xl, xb - xl, xc - xl, xd - xl]
        
        x0_cat = torch.cat([xa, xb, xc, xd], dim=1)
        xh_cat = torch.cat(xh, dim=1)
        
        xh_enhanced = self.afs1(xh_cat)
        x0_enhanced = self.afs2(x0_cat)
        
        x_enhanced = torch.cat([xh_enhanced, x0_enhanced], dim=1)
        x_enhanced = self.conv1(x_enhanced)
        x_enhanced = self.ca_layer(x_enhanced)
        x_enhanced = self.afs3(x_enhanced)
        
        out = self.sigma * x0_cat + x_enhanced
        out = self.conv_final(out)
        
        return out
