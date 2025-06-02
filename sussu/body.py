import torch
import torch.nn as nn

from attention import LocalAttention
from potential import Potential
from memory import Memory


class sim_body(nn.Module):
    def __init__(self, size, hidden_size=64):
        super(sim_body, self).__init__()
        self.transor = nn.Linear(size, hidden_size)
        self.memory = Memory(hidden_size)
        self.thought = LocalAttention(hidden_size)

        # 其他组件保留不变
        self.fit = nn.Linear(size + hidden_size, size)
        self.lim = nn.LayerNorm(size)
        self.activate = nn.GELU()

    def forward(self, x):
        x = self.transor(x)
        
        mind = self.memory(x)
        mind = self.thought(mind)
        
        # 后续处理保持一致
        x = self.fit(torch.cat([x, mind], dim=-1))
        x = self.lim(x)
        output = self.activate(x)

        return output

        
class Body(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=32, scale=8):
        super(Body, self).__init__()
        self.fit_in = nn.Linear(in_size, hidden_size)
        self.fit_out = nn.Linear(hidden_size, out_size)
        
        self.body = nn.ModuleList([
            sim_body(hidden_size, hidden_size)
            for _ in range(scale)
        ])

    def forward(self, x):
        x = self.fit_in(x)
        for body in self.body:
            x = body(x)

        output = self.fit_out(x)
        return output