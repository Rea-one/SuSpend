import torch
import torch.nn as nn

from attention import LocalAttention
from potential import Potential
from memory import Memory
from linker import Linker


class sim_body(nn.Module):
    def __init__(self, size, activate_choice, max_choice):
        super(sim_body, self).__init__()
        self.linker = Linker(size, max_choice, choices=activate_choice)
        self.map = nn.ModuleList([Memory(size) for _ in range(max_choice)])
        self.thought = nn.ModuleList([LocalAttention(size) for _ in range(max_choice)])

        # 其他组件保留不变
        self.fit = nn.Linear(size * (activate_choice + 1), size)
        self.lim = nn.LayerNorm(size)
        self.activate = nn.GELU()

    def forward(self, x, choices=2, scores=None):
        batch_size, seq_len, _ = x.shape
        
        for b in range(batch_size):
            for seq_idx in range(seq_len):
                for choice_idx in range(choices):
                    x[b, seq_idx, :] += self.lim(self.map[choice_idx](x[b, seq_idx, :]))

        x = torch.cat([self.linker(x), x], dim=-1)
                    
        # 后续处理保持一致
        x = self.fit(x)
        x = self.lim(x)
        output = self.activate(x)

        return output


class Body(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=32, activate_choice=2, max_choice=32, scale=1):
        super(Body, self).__init__()
        self.fit_in = nn.Linear(in_size, hidden_size)
        self.fit_out = nn.Linear(hidden_size, out_size)
        
        self.body = nn.ModuleList([
            sim_body(hidden_size, activate_choice, max_choice)
            for _ in range(scale)
        ])

    def forward(self, x):
        x = self.fit_in(x)
        for body in self.body:
            x = body(x)

        output = self.fit_out(x)
        return output