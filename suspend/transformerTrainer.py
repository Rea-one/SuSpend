# transformerTrainer_datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset

# ================== 1. 自定义 Dataset ==================
class HFDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer=None, vocab=None, max_length=128):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer if tokenizer else lambda x: x.split()
        self.max_length = max_length

        # 构建标签编码器
        self.label_encoder = LabelEncoder()
        self.labels = [item['label'] for item in hf_dataset]
        self.label_encoder.fit(self.labels)

        # 构建词汇表或传入外部词汇表
        if vocab is None:
            self.vocab = self.build_vocab([self.tokenizer(item['text']) for item in hf_dataset])
        else:
            self.vocab = vocab

    def build_vocab(self, token_sequences):
        vocab = {}
        idx = 0
        for tokens in token_sequences:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
        vocab["<unk>"] = idx  # 添加 <unk> 标记
        return vocab

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        tokens = self.tokenizer(item['text'])
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens[:self.max_length]]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        label_tensor = torch.tensor(self.label_encoder.transform([item['label']])[0], dtype=torch.long)
        length_tensor = torch.tensor(len(token_tensor), dtype=torch.int64)
        return token_tensor, label_tensor, length_tensor


# ================== 2. 自定义 collate 函数 ==================
def collate_batch(batch):
    texts, labels, lengths = zip(*batch)
    texts = pad_sequence(texts, batch_first=False, padding_value=0)
    labels = torch.stack(labels)
    lengths = torch.stack(lengths)
    return texts, labels, lengths


# ================== 3. Transformer 模型封装 ==================
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


# ================== 4. 加载 HuggingFace 数据集（IMDB）==================
dataset = load_dataset("imdb")

# 取部分数据用于快速训练（可选）
train_dataset = HFDataset(dataset['train'].shuffle(seed=42).select(range(10000)))
val_dataset = HFDataset(dataset['test'].shuffle(seed=42).select(range(2000)))

# 创建 DataLoader
BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

vocab_size = len(train_dataset.vocab)
OUTPUT_SIZE = len(set([d['label'] for d in dataset['train']]))  # 自动推断类别数


# ================== 5. 初始化模型、优化器、损失函数 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(vocab_size, embed_dim=32, hidden_size=64, output_size=OUTPUT_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# ================== 6. 训练函数 ==================
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels, _ in dataloader:
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


# ================== 7. 验证函数 ==================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels, _ in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Val Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


# ================== 8. 开始训练 ==================
EPOCHS = 5
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)


# ================== 9. 保存模型 ==================
torch.save(model.state_dict(), "transformer_imdb_model.pth")
print("Transformer 模型已保存！")