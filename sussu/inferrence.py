# inference.py
import torch
from suspend import SuspendStack
from config import get_args
from tiktoken import get_encoding


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    args = get_args()

    # 加载词表编码器
    enc = get_encoding("cl100k_base")
    vocab_size = enc.n_vocab

    # 初始化模型
    model = SuspendStack(voc_size=vocab_size, output_size=vocab_size).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    return model, enc


def generate_text(model, enc, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    """使用模型生成文本"""
    encode = lambda s: enc.encode(s, allowed_special={"", })
    decode = lambda l: enc.decode(l)

    start_ids = encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        generated_text = decode(y[0].tolist())

    return generated_text


if __name__ == "__main__":
    # 模型路径和设备设置
    checkpoint_path = "checkpoints/suspend_model_epoch_5.pth"  # 替换为你的模型路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model, enc = load_model(checkpoint_path, device)

    # 输入提示
    prompt = "什么是递归呢？"

    # 生成文本
    generated_text = generate_text(model, enc, prompt)
    print(generated_text)