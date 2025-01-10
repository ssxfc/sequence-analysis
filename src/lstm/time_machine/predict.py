import time
import torch
import torch.utils

from torch import nn
from torch.nn import functional as F

import utils
import model


def generate(net:nn.Module, fp, text_head):
    r"""文本生成"""
    with open(fp, "r", encoding="utf-8") as f:
        str_labels = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    chars = ["<unk>"] + sorted([char for char in set(str_labels.lower())])
    char_map = {token: idx for idx, token in enumerate(chars)}

    net.eval()
    state = None
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    text = text_head.lower()
    print(text, end="")
    with torch.no_grad():
        # 预测
        while True:
            x = [char_map[char] for char in text]
            x = torch.as_tensor(x).unsqueeze(dim=0).to(device)
            state = net.begin_state(batch_size=1, device=device)
            output, state = net(x, state)

            next_pred_char = chars[F.softmax(output[-1], dim=0).argmax(dim=0).item()]
            text = text[1:] + next_pred_char
            print(next_pred_char, end="")
            time.sleep(1)


def infer1(net:nn.Module, fp, text_head, num):
    r"""序列预测"""
    with open(fp, "r", encoding="utf-8") as f:
        str_labels = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    chars = ["<unk>"] + sorted([char for char in set(str_labels.lower())])
    char_map = {token: idx for idx, token in enumerate(chars)}
    # 设置模型工作模式，以防止某些module出现非预期行为
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    state = net.begin_state(batch_size=1, device=device)
    text = text_head.lower()
    all_text = text
    print(text, end="")
    # 禁用梯度计算，节省资源消耗
    with torch.no_grad():
        x = [char_map[char] for char in text]
        x = torch.as_tensor(x).unsqueeze(dim=0).to(device)
        output, state = net(x, state)
        pred_y = "".join(chars[F.softmax(output[-1], dim=0).argmax(dim=0)])
        all_text += pred_y
        for i in range(num):
            x = [char_map[all_text[-1]]]
            x = torch.as_tensor(x).unsqueeze(dim=0).to(device)
            output, state = net(x, state)
            pred_y = "".join(chars[F.softmax(output[-1], dim=0).argmax(dim=0)])
            all_text = pred_y
            print(pred_y[-1], end="")
            time.sleep(0.5)
        print()


def infer2(net:nn.Module, fp, text_head, num):
    r"""序列预测"""
    with open(fp, "r", encoding="utf-8") as f:
        str_labels = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    chars = ["<unk>"] + sorted([char for char in set(str_labels.lower())])
    char_map = {token: idx for idx, token in enumerate(chars)}
    # 设置模型工作模式，以防止某些module出现非预期行为
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    state = net.begin_state(batch_size=1, device=device)
    text = text_head.lower()
    # print(text, end="")
    # 禁用梯度计算，节省资源消耗
    with torch.no_grad():
        outputs = [char_map[text[0]]]                                       
        get_inputs = lambda: torch.tensor([outputs[-1]], device=device).reshape(1, 1)
        for y in text[1:]:                                                       
            _, state = net(get_inputs(), state)                                    
            outputs.append(char_map[y])                                       
        for _ in range(num):                                                 
            y, state = net(get_inputs(), state)                                    
            outputs.append(int(y.argmax(dim=1).reshape(1)))
        print("".join(chars[i] for i in outputs))                        
        return "".join(chars[i] for i in outputs)


if __name__ == "__main__":
    utils.set_seed(37)
    args = {
        "num_epochs": 50,
        "batch_size": 16,
        "lr": 0.0001,
        "num_steps": 32,
        "num_hiddens": 1024,
        "num_layers": 1,
        "fp": r"/home/dcd/zww/repos/sequence-analysis/data/labels.txt"
    }
    with open(args['fp'], "r", encoding="utf-8") as f:
        str_labels = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    chars = ["<unk>"] + sorted([char for char in set(str_labels.lower())])
    char_map = {token: idx for idx, token in enumerate(chars)}
    # lstm前馈
    # layer = nn.GRU(input_size=len(chars), hidden_size=args["num_hiddens"], num_layers=args["num_layers"])
    layer = nn.LSTM(input_size=len(chars), hidden_size=args["num_hiddens"], num_layers=args["num_layers"])
    net = model.RNNModel(layer, vocab_size=len(chars))
    net.load_state_dict(torch.load("checkpoints/model_lstm300.pth", map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
    header = "describe how it relieved me to think that"
    infer2(net, args['fp'], header, 60)
