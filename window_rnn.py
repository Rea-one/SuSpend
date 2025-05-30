import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowedSelfAttention(nn.Module):
    """
    带局部窗口的自注意力模块。
    只允许当前 token 关注前面 fixed_window_size 个 token。
    """
    def __init__(self, n_embd, n_head, fixed_window_size=16, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.fixed_window_size = fixed_window_size
        self.head_dim = n_embd // n_head
        
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)

        # 预先生成一个局部窗口掩码
        self.register_buffer('mask', None)

    def forward(self, x):
        B, T, C = x.size()
        
        # 生成 q, k, v
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.n_head, self.head_dim).transpose(1, 2), qkv)

        # 计算 attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 构建局部窗口掩码
        if self.mask is None or self.mask.shape != (1, 1, T, T):
            mask = torch.ones(T, T).tril()  # 下三角矩阵
            mask_cond = torch.arange(T).unsqueeze(0) - torch.arange(T).unsqueeze(1) >= self.fixed_window_size
            mask.masked_fill_(mask_cond, 0)
            self.mask = mask.unsqueeze(0).unsqueeze(0).to(att.device)

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y


class WindowRNNBlock(nn.Module):
    """
    类似 RNN 的模块，每一步使用局部窗口 attention。
    支持缓存 key/value，以便高效推理。
    """
    def __init__(self, n_embd, n_head, fixed_window_size=16):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = WindowedSelfAttention(n_embd, n_head, fixed_window_size=fixed_window_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x, cache=None):
        # cache 用于保存 key/value，提升推理效率
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class WindowRNNModel(nn.Module):
    """
    完整的 WindowRNN 模型。
    支持逐 token 生成和缓存 key/value。
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, fixed_window_size=16):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(1024, n_embd)  # 简单位置编码
        self.h = nn.ModuleList([
            WindowRNNBlock(n_embd, n_head, fixed_window_size=fixed_window_size)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None, cache=None):
        device = idx.device
        B, T = idx.size()
        pos = torch.arange(T, dtype=torch.long, device=device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb

        for block in self.h:
            x = block(x, cache=cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        逐 token 生成，支持缓存 key/value 加速推理。
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