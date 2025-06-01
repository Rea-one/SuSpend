# utils.py
import os
import torch


def save_checkpoint(state, save_path):
    """保存模型和 optimizer 状态"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    print(f"Checkpoint saved at {save_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """加载模型、optimizer 和 scheduler 的状态"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from epoch {start_epoch}")
    return start_epoch


class EarlyStopping:
    def __init__(self, patience=3, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss if self.mode == 'min' else val_loss

        if self.best_score is None:
            self.best_score = score
        elif (score < self.best_score + self.delta and self.mode == 'min') or \
                (score > self.best_score - self.delta and self.mode == 'max'):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0