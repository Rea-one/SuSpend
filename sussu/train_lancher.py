# train_launcher.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import tiktoken
from tqdm import tqdm, trange

# 自定义模块
from suspend import SuspendStack
from dataset import BinDataset
from config import get_args
from logger import TensorBoardLogger
from utils import save_checkpoint, load_checkpoint, EarlyStopping


def main():
    args = get_args()

    # 设备设置
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据集
    vocab_size = tiktoken.get_encoding("cl100k_base").n_vocab
    OUTPUT_SIZE = vocab_size

    train_dataset = BinDataset('data/train.bin', block_size=128)
    val_dataset = BinDataset('data/val.bin', block_size=128)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # 模型
    model = SuspendStack(voc_size=vocab_size, output_size=OUTPUT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, None, args.resume)

    # 学习率调度器
    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=args.eta_min)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()

    # 日志与早停
    logger = TensorBoardLogger()
    early_stopping = EarlyStopping(patience=args.patience) if args.early_stop else None

    # 训练循环
    for epoch in trange(start_epoch, args.epochs, desc="Training"):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            # 指标统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total_samples += y.numel()

            # 更新进度条
            avg_loss = total_loss / (progress_bar.n + 1)
            accuracy = correct / total_samples
            progress_bar.set_postfix(loss=avg_loss, acc=accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += y.numel()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # 日志记录
        logger.log_scalar("Loss/train", avg_train_loss, epoch)
        logger.log_scalar("Loss/val", avg_val_loss, epoch)
        logger.log_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存 checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': avg_val_loss,
        }, os.path.join(args.save_dir, f"suspend_model_epoch_{epoch+1}.pth"))

        # 早停判断
        if early_stopping:
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
                
    logger.close()
    

if __name__ == "__main__":
    main()