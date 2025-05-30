import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAttention(nn.Module):
    """
    局部窗口 attention，只允许当前 token 关注前面 fixed_window_size 个 token。
    """
    def __init__(self, n_embd, fixed_window_size=16):
        super().__init__()
        self.fixed_window_size = fixed_window_size
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)  # [B, T, C]
        v = self.v_proj(x)  # [B, T, C]

        # 构造局部窗口掩码
        mask = torch.ones(T, T).triu_(diagonal=-self.fixed_window_size).bool().to(x.device)
        
        # 计算 attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C))
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = self.out_proj(y)
        return y


class RNNWithLocalAttention(nn.Module):
    """
    使用 LSTM + 局部 attention 的类 RNN 模型。
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, fixed_window_size=16):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.attn = LocalAttention(hidden_size, fixed_window_size=fixed_window_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, targets=None):
        x = self.wte(idx)
        x, _ = self.rnn(x)
        x = self.ln(x + self.attn(x))
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        逐 token 生成，支持局部 attention。
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 1024 else idx[:, -1024:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx