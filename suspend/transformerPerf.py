# transformerPerf.py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn

# ================== 1. 模型定义 ==================
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_layer = nn.Linear(embed_dim, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x)
        return self.output_layer(out.mean(dim=0))


# ================== 2. 加载模型 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(10000, 32, 64, 4).to(device)
model.load_state_dict(torch.load("transformer_model.pth"))
model.eval()

# ================== 3. 测试推理 ==================
test_iter = AG_NEWS(split="test")
test_loader = DataLoader(list(test_iter)[:100], batch_size=16, collate_fn=collate_batch)

correct = 0
total = 0

with torch.no_grad():
    for texts, labels, _ in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")