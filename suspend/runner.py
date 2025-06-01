# runner.py

import os
import torch
import numpy as np
import tiktoken

from suspend import Suspend
from attention import LocalAttention


def load_model(model_path, voc_size, output_size):
    """
    加载训练好的模型权重
    :param model_path: 模型权重路径
    :param voc_size: 词汇表大小
    :param output_size: 输出维度
    :return: 加载后的模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Suspend(voc_size=voc_size, output_size=output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    print(f"模型已加载：{model_path}")
    return model


def predict(model, input_seq):
    """
    使用模型进行预测
    :param model: 训练好的模型
    :param input_seq: 输入序列 (tensor)
    :return: 预测结果
    """
    device = next(model.parameters()).device  # 获取模型所在设备
    input_seq = input_seq.to(device)

    with torch.no_grad():
        logits = model(input_seq)
        predictions = torch.argmax(logits, dim=-1)

    return predictions.cpu().numpy()


def prepare_input(text, encoder):
    """
    将输入文本编码为 token ID 序列
    :param text: 输入文本
    :param encoder: 编码器
    :return: tensor 格式的 token ID 序列
    """
    tokens = encoder.encode(text)
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # 添加 batch 维度
    return input_tensor


def main():
    # 参数设置
    model_path = "suspend_imdb_model.pth"
    vocab_size = tiktoken.get_encoding("cl100k_base").n_vocab
    output_size = vocab_size

    # 加载模型
    model = load_model(model_path, vocab_size, output_size)

    # 初始化编码器
    encoder = tiktoken.get_encoding("cl100k_base")

    # 示例输入
    input_text = "This is a test input for the Suspend model."
    input_seq = prepare_input(input_text, encoder)

    # 进行预测
    predictions = predict(model, input_seq)

    # 解码预测结果
    predicted_text = encoder.decode(predictions[0])
    print(f"输入文本：{input_text}")
    print(f"预测结果：{predicted_text}")


if __name__ == "__main__":
    main()