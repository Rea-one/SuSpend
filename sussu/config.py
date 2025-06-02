# config.py
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Train Suspend Model")

    # 基础参数
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--use_cuda', action='store_true', default=True)

    # 学习率调度
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--eta_min', type=float, default=1e-6)

    # 早停机制
    parser.add_argument('--early_stop', action='store_true', default=True)
    parser.add_argument('--patience', type=int, default=3)

    return parser.parse_args()