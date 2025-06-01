# dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset


class BinDataset(Dataset):
    def __init__(self, data_path, block_size, cursor_size=16):
        self.data = np.fromfile(data_path, dtype=np.uint32)
        self.block_size = block_size
        self.cursor_size = cursor_size

    def __len__(self):
        return len(self.data) // self.block_size
        
    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.tensor(self.data[start:end], dtype=torch.long)

        pad_len = (self.cursor_size - (x.size(0) % self.cursor_size)) % self.cursor_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, pad_len), value=0)

        y = torch.tensor(self.data[start+1:end+1], dtype=torch.long)
        if pad_len > 0:
            y = torch.nn.functional.pad(y, (0, pad_len), value=0)

        return x, y