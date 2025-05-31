import torch
import torch.nn as nn


class LocalAttention(nn.Module):
    def __init__(self, n_embd, fixed_window_size=16):
        super().__init__()
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.fixed_window_size = fixed_window_size
        self.n_embd = n_embd

    def forward(self, x):
        """
        x: 输入张量，形状为 (B, T, C)，即 (batch_size, seq_len, embed_dim)
        """
        B, T, C = x.size()

        # 线性变换生成 QKV
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # 初始化输出
        output = []

        for t in range(T):
            # 确定当前 token 可见的局部窗口范围
            start = max(0, t - self.fixed_window_size // 2)
            end = min(T, t + self.fixed_window_size // 2 + 1)

            # 提取局部 key 和 value
            local_k = k[:, start:end, :]  # (B, W, C)
            local_v = v[:, start:end, :]  # (B, W, C)

            # 获取当前时间步的 query 向量
            current_q = q[:, t:t+1, :]  # (B, 1, C)

            # 计算注意力得分
            attn_weights = (current_q @ local_k.transpose(-2, -1)) / (C ** 0.5)  # (B, 1, W)
            attn_weights = attn_weights.softmax(dim=-1)  # 归一化

            # 加权求和得到上下文向量
            context = attn_weights @ local_v  # (B, 1, C)

            # 输出拼接
            output.append(context)

        # 拼接所有时间步的输出
        output = torch.cat(output, dim=1)  # (B, T, C)
        
        # 最后通过输出投影层
        output = self.out_proj(output)
        
        return output