import torch
import torch.nn as nn

from torch.nn import functional as F

from attention import LocalAttention

class Suspend(nn.Module):
    def __init__(self, voc_size, output_size, hidden_size=64, cursor_size = 16):
        super(Suspend, self).__init__()
        
        self.sus = None
        
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
        if self.sus is None:
            self.sus = sus
        
        if self.sus is None and sus is None:
            self.sus = torch.zeros(batch_size, self.cursor_size, self.hidden_size).to(x.device)

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

            combined = torch.cat([watch, self.sus], dim=-1)
            
            self.sus = self.activate(self.suspend(combined))
            self.sus = self.doubt(self.sus)

            # 输出
            output = self.determin(combined)
            output = self.check(output)
            outputs.append(output)

        # 拼接所有输出
        return torch.cat(outputs, dim=1)

class SuS(nn.Module):
    def __init__(self, output_size, hidden_size=64, cursor_size = 16):
        super(SuS, self).__init__()
        
        self.sus = None
        
        self.cursor_size = cursor_size
        self.hidden_size = hidden_size
        
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

        batch_size, seq_len, _= x.size()
        
        if self.sus is None:
            self.sus = sus
        
        if self.sus is None and sus is None:
            self.sus = torch.zeros(batch_size, self.cursor_size, self.hidden_size).to(x.device)
            
            
        outputs = []

        for cursor in range(0, seq_len, self.cursor_size):
            end = min(cursor + self.cursor_size, seq_len)
            read = x[:, cursor:end, :]
            
            # 注意力机制
            watch = self.attention(read)
            watch = self.lim(watch)
            watch = self.dropout(watch)

            combined = torch.cat([watch, self.sus], dim=-1)
            
            self.sus = self.activate(self.suspend(combined))
            self.sus = self.doubt(self.sus)

            # 输出
            output = self.determin(combined)
            output = self.check(output)
            outputs.append(output)

        # 拼接所有输出
        return torch.cat(outputs, dim=1)
        
class SuspendBlock(nn.Module):
    def __init__(self, hidden_size=128, cursor_size=32):
        super(SuspendBlock, self).__init__()
        self.norm = nn.LayerNorm(cursor_size)
        self.suspend = SuS(output_size=cursor_size, hidden_size=hidden_size, cursor_size=cursor_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, sus=None):
        residual = x
        x = self.suspend(x, sus=sus)
        x = self.norm(x)
        x = self.dropout(x)
        x += residual  # 残差连接
        return x

        
        
class SuspendStack(nn.Module):
    def __init__(self, voc_size, output_size, num_layers=4, hidden_size=128, cursor_size=32):
        super(SuspendStack, self).__init__()
        
        self.head = Suspend(voc_size, output_size)
        
        self.layers = num_layers
        
        # 构建多个 SuspendBlock
        self.blocks = nn.ModuleList([
            SuspendBlock(hidden_size=hidden_size, cursor_size=cursor_size)
            for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Linear(hidden_size, output_size)
        self.ln_f = nn.LayerNorm(output_size)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_size, seq_len = x.size()
        
        # 嵌入层
        x = self.head(x)
        
        # 多层 SuspendBlock 堆叠
        for block in self.blocks:
            x = block(x)
        
        # 最终输出层
        x = self.final_layer(x)
        x = self.ln_f(x)
        
        return x
