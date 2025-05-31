import torch
import torch.nn as nn

from attention import LocalAttention


class Suspend(nn.Module):
    def __init__(self, hidden_size, output_size, cursor_size = 16):
        super(Suspend, self).__init__()
        
        self.cursor_size = cursor_size
        
        self.attention = LocalAttention(self.cursor_size)
        self.suspend = nn.Linear(hidden_size + self.cursor_size, hidden_size)
        self.determin = nn.Linear(cursor_size, output_size)
        # 激活函数
        self.tanh = nn.Tanh()
        
    def forward(self, x, sus=None):
        """
        x: 输入序列，形状为 (seq_len, batch_size, input_size)
        """
        seq_len, batch_size, _ = x.size()
        hidden_size = self.inside.weight.shape[0]

        # 初始化悬置体
        if sus is None:
            sus = torch.zeros(seq_len, batch_size, hidden_size + self.cursor_size).to(x.device)
        outputs = []
        for cursor in range(0, seq_len, self.cursor_size):
            
            # 初始版
            read = x[cursor:  cursor + self.cursor_size]
            watch = self.attention(read)
            sus = self.tanh(self.suspend(watch + sus))
            output = self.determin(watch)
            outputs.append(output)
            
        return torch.stack(outputs)
        