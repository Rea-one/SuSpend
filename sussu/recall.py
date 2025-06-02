import torch
import torch.nn as nn

from attention import LocalAttention

class Recall(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.attention = LocalAttention(size)
        self.lim = nn.LayerNorm(size)
        self.activate = nn.GELU()
        self.passer = None

    def forward(self, x):
        if self.passer is None:
            self.passer = torch.zeros_like(x)
        x = torch.cat([x, self.passer], dim=1)
        x = self.attention(x)
        x = self.lim(x)
        x = self.activate(x)
        return x
