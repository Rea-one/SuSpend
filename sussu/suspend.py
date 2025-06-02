import torch
import torch.nn as nn

from torch.nn import functional as F

from attention import LocalAttention

from body import Body

class Sus(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(Sus, self).__init__()
        
        self.sus = None
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.attention = LocalAttention(input_size)
        self.lim = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(p=0.5)
        self.suspend = nn.Linear(hidden_size + input_size, hidden_size)
        self.doubt = nn.LayerNorm(hidden_size)
        self.determin = nn.Linear(hidden_size + input_size, output_size)
        self.check = nn.LayerNorm(output_size)
        # 激活函数
        self.activate = nn.GELU()
        
    def forward(self, x, sus=None):        
        
        # 初始化悬置体
        if self.sus is None:
            self.sus = sus
        
        if self.sus is None and sus is None:
            batch_size, seq_len, voc_size = x.shape
            self.sus = torch.zeros(batch_size, seq_len, self.hidden_size).to(x.device)
            
        
        # 注意力机制
        watch = self.attention(x)
        watch = self.lim(watch)
        watch = self.dropout(watch)

        combined = torch.cat([self.sus, watch], dim=-1)
        
        self.sus = self.activate(self.suspend(combined))
        self.sus = self.doubt(self.sus)

        # 输出
        output = self.determin(combined)
        output = self.check(output)

        # 拼接所有输出
        return output  
        

class SusTail(nn.Module):
    def __init__(self, in_size, out_size):
        super(SusTail, self).__init__()
        self.fit = nn.Linear(in_size, out_size)
        self.lim = nn.LayerNorm(out_size)
        self.activate = nn.GELU()
    
    def forward(self, x):
        return self.activate(self.lim(self.fit(x)))

        
class SuspendStack(nn.Module):
    def __init__(self, voc_size, output_size, scale=4, hidden_size=64, cursor_size=16):
        super(SuspendStack, self).__init__()
        
        self.cursor_size = cursor_size
        
        self.embedding = nn.Embedding(voc_size, cursor_size)
        
        self.head = Sus(cursor_size, hidden_size, hidden_size)
        
        self.scale = scale
        
        self.body = Body(hidden_size, hidden_size, hidden_size=16, scale=scale)
        
        self.tail = SusTail(hidden_size, output_size)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        outputs = []
        
        for order in range(0, seq_len, self.cursor_size):
            window = x[:, order:order + self.cursor_size]
            
            read = self.embedding(window)
            watch = self.head(read)
            ana = self.body(watch)
            output = self.tail(ana)
            outputs.append(output)
            
        return torch.cat(outputs, dim=1)