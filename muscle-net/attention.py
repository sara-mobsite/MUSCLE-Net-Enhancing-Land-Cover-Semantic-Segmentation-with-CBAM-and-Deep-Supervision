import torch
from torch import nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.fc1 = nn.Linear(in_channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, in_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()

        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, channels)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool))).view(batch_size, channels, 1, 1)
        max_out = self.fc2(F.relu(self.fc1(max_pool))).view(batch_size, channels, 1, 1)

        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_pool, max_pool], dim=1)
        attention = torch.sigmoid(self.conv(x_cat))
        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
