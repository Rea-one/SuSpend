# RNNperf.py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import AG_NEWS
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn

# ================== 1. 数据预处理 ==================
def collate_batch(batch):
    label_pipeline = lambda x: int(x) - 1
    text_pipeline = lambda x: x.split()

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield text_pipeline(text)

    vocab = build_vocab_from_iterator(yield_tokens(AG_NEWS(split="train")),
                                      specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    label_list, text_list, length_list = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(vocab(text_pipeline(_text)), dtype=torch.long)
        text_list.append(processed_text)
        length_list.append(processed_text.shape[0])

    label_tensor = torch.tensor(label_list, dtype=torch.long)
    text_tensor = pad_sequence(text_list, batch_first=False)
    length_tensor = torch.tensor(length_list, dtype=torch.int64)

    return text_tensor, label_tensor, length_tensor


# ================== 2. 模型定义 ==================
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=False)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.output_layer(out.mean(dim=0))


# ================== 3. 加载模型 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel(10000, 32, 64, 4).to(device)
model.load_state_dict(torch.load("rnn_model.pth"))
model.eval()

# ================== 4. 测试推理 ==================
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