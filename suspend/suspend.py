import torch
import torch.nn as nn

from torch.nn import functional as F

from attention import LocalAttention


class Suspend(nn.Module):
    def __init__(self, voc_size, output_size, hidden_size=64, cursor_size = 16):
        super(Suspend, self).__init__()
        
        self.cursor_size = cursor_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(voc_size, cursor_size)
        
        self.attention = LocalAttention(cursor_size)
        self.lim = nn.LayerNorm(cursor_size)
        self.dropout = nn.Dropout(p=0.5)
        self.suspend = nn.Linear(hidden_size + cursor_size, hidden_size)
        self.doubt = nn.LayerNorm(hidden_size)
        self.determin = nn.Linear(hidden_size + cursor_size, output_size)
        self.check = nn.LayerNorm(output_size)
        # 激活函数
        self.activate = nn.GELU()
        
    def forward(self, x, sus=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size, seq_len = x.size()
        
        # 初始化悬置体
        if sus is None:
            sus = torch.zeros(batch_size, self.cursor_size, self.hidden_size).to(x.device)

        outputs = []

        for cursor in range(0, seq_len, self.cursor_size):
            end = min(cursor + self.cursor_size, seq_len)
            current_x = x[:, cursor:end]

            # 嵌入层
            read = self.embedding(current_x)
            
            # 注意力机制
            watch = self.attention(read)
            watch = self.lim(watch)
            watch = self.dropout(watch)

            combined = torch.cat([watch, sus], dim=-1)
            
            sus = self.activate(self.suspend(combined))
            sus = self.doubt(sus)

            # 输出
            output = self.determin(combined)
            output = self.check(output)
            outputs.append(output)

        # 拼接所有输出
        return torch.cat(outputs, dim=1)
        