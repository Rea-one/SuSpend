# logger.py
from torch.utils.tensorboard import SummaryWriter
import os


class TensorBoardLogger:
    def __init__(self, log_dir="tb_logs"):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()