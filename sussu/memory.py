import torch
import torch.nn as nn



class Memory(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.in_filter = nn.Linear(size, 1)
        self.memory = nn.Linear(size + 1, size)
        self.out_filter = nn.Linear(size, 1)
        self.guard = nn.Linear(size + 1, size)
        self.lim = nn.GELU()
        
    def forward(self, x):
        the_in = torch.cat([x, self.in_filter(x)], dim=-1)
        mind = self.memory(the_in)
        the_out = torch.cat([mind, self.out_filter(mind)], dim=-1)
        output = self.lim(self.guard(the_out))
        return output