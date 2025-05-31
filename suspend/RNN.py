import torch
import torch.nn as nn

class MinimalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MinimalRNN, self).__init__()
        # 输入到隐藏层的权重矩阵
        self.W_xh = nn.Linear(input_size, hidden_size)
        # 隐藏层到隐藏层的权重矩阵（循环连接）
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        # 隐藏层到输出的权重矩阵
        self.W_out = nn.Linear(hidden_size, output_size)
        
        # 激活函数
        self.tanh = nn.Tanh()

    def forward(self, x, hidden=None):
        """
        x: 输入序列，形状为 (seq_len, batch_size, input_size)
        """
        seq_len, batch_size, _ = x.size()
        hidden_size = self.W_hh.weight.shape[0]

        # 初始化隐藏状态
        if hidden is None:
            hidden = torch.zeros(batch_size, hidden_size)

        outputs = []
        for t in range(seq_len):
            # 当前时间步输入
            xt = x[t]
            # 更新隐藏状态：h_t = tanh(W_xh * x_t + W_hh * h_{t-1})
            hidden = self.tanh(self.W_xh(xt) + self.W_hh(hidden))
            # 输出
            out = self.W_out(hidden)
            outputs.append(out.unsqueeze(0))  # 添加时间步维度

        # 将输出拼接成完整序列
        return torch.cat(outputs, dim=0), hidden