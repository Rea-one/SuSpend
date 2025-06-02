import torch
import torch.nn as nn
from torch.nn import functional as F



class Linker(nn.Module):
    def __init__(self, input_dim, ceil, choices=1):
        super(Linker, self).__init__()
        self.router = nn.Linear(input_dim, ceil)
        self.choices = choices

    def forward(self, x):
        logits = self.router(x)  # (B, T, E)
        scores = F.softmax(logits, dim=-1)  # 转换为概率分布
        choices_scores, choices_indices = torch.topk(scores, self.choices, dim=-1)  # (B, T, K), (B, T, K)

        return choices_scores, choices_indices, scores
        

def balance_loss(probs):
    avg_probs = probs.mean(dim=0)
    uniform = torch.ones_like(avg_probs) / avg_probs.size(0)
    loss = F.kl_div(avg_probs.log(), uniform, reduction='batchmean')
    return loss