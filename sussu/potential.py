import torch
import torch.nn as nn

from attention import LocalAttention



class Potential(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.attention = LocalAttention(size)
        self.random = nn.Linear(size, size)
        self.lim = nn.LayerNorm(size)
        self.activate = nn.GELU()
        
    def forward(self, x):
        read = self.attention(x)
        sence = self.random(x)
        
        read = self.lim(read)
        sence = self.lim(sence)
        
        combine = read + sence
        
        output = self.activate(combine)
        
        return output