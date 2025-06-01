import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import tiktoken
import tqdm
from tqdm import trange

from suspend import Suspend

# ================== 数据集定义 ==================
class BinDataset(Dataset):
    def __init__(self, data_path, block_size, cursor_size=16):
        self.data = np.fromfile(data_path, dtype=np.uint32)
        self.block_size = block_size
        self.cursor_size = cursor_size

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.tensor(self.data[start:end], dtype=torch.long)

        # 自动填充到 cursor_size 的整数倍
        pad_len = (self.cursor_size - (x.size(0) % self.cursor_size)) % self.cursor_size
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=0)  # 假设 0 是 <pad> token

        y = torch.tensor(self.data[start+1:end+1], dtype=torch.long)
        if pad_len > 0:
            y = F.pad(y, (0, pad_len), value=0)

        return x, y


BATCH_SIZE = 1
BLOCK_SIZE = 128

# 加载训练和验证数据集
train_dataset = BinDataset('data/train.bin', BLOCK_SIZE)
val_dataset = BinDataset('data/val.bin', BLOCK_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

vocab_size = tiktoken.get_encoding("cl100k_base").n_vocab
OUTPUT_SIZE = vocab_size


# ================== 初始化模型、优化器、损失函数 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Suspend(voc_size=vocab_size, output_size=OUTPUT_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ================== 添加模型保存相关配置 ==================
save_interval = 1  # 每隔几个 epoch 保存一次模型
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)


# ================== 训练函数 ==================
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 使用 tqdm 包裹 dataloader
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # 更新正确预测的数量
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == y).sum().item()
        total += y.numel()

        # 实时更新进度条描述
        avg_loss = total_loss / len(progress_bar)
        accuracy = correct / total
        progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

    
# ================== 验证函数 ==================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            total_loss += loss.item()
            
            # 更新正确预测的数量
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == y).sum().item()
            total += y.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Val Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
    

# ================== 开始训练 ==================
EPOCHS = 5
for epoch in range(EPOCHS):
    print(f"\训练进度 {epoch+1}/{EPOCHS}")
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    # ================== 添加模型保存逻辑 ==================
    if (epoch + 1) % save_interval == 0:
        model_save_path = os.path.join(save_dir, f"suspend_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"第 {epoch+1} 轮训练完成，模型已保存到 {model_save_path}")


# ================== 保存最终模型 ==================
torch.save(model.state_dict(), "Suspend_final.pth")
print("最终 Suspend 模型已保存！")