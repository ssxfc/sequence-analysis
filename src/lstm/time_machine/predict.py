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
    all_txt = text
    with torch.no_grad():
        # 预测
        while True:
            x = [char_map[char] for char in text]
            x = torch.as_tensor(x).unsqueeze(dim=0).to(device)
            state = net.begin_state(batch_size=1, device=device)
            output, state = net(x, state)

            next_pred_char = chars[F.softmax(output[-1], dim=0).argmax(dim=0).item()]
            all_txt += next_pred_char
            text = text[1:] + next_pred_char
            print(all_txt)
            time.sleep(1)

def infer(net:nn.Module, fp, text_head):
    r"""序列预测
    缺陷：
        根据一个已有序列进行预测，只能得到1个预测字符，正确预测概率假设为0.99
        那第二个字符也预测正确的概率为0.99^2
        仅仅预测五十个单词，全部成功概率为0.6左右
    """
    with open(fp, "r", encoding="utf-8") as f:
        str_labels = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    chars = ["<unk>"] + sorted([char for char in set(str_labels.lower())])
    char_map = {token: idx for idx, token in enumerate(chars)}

    net.eval()
    state = None
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    text = text_head.lower()
    all_txt = text
    with torch.no_grad():
        for i in range(100):
            x = [char_map[char] for char in text]
            x = torch.as_tensor(x).unsqueeze(dim=0).to(device)
            state = net.begin_state(batch_size=1, device=device)
            output, state = net(x, state)
            print("")
            pred_y = "".join([chars[id] for id in F.softmax(output, dim=1).argmax(dim=1)])
            print(pred_y)
            all_txt += pred_y[-1]
            text = pred_y
        print(all_txt)


if __name__ == "__main__":
    utils.set_seed(37)
    args = {
        "num_epochs": 20,
        "batch_size": 1,
        "lr": 0.0001,
        "num_steps": 64,
        "num_hiddens": 1024,
        "num_layers": 1,
        "fp": r"/home/dcd/zww/repos/sequence-analysis/data/labels.txt"
    }
    with open(args['fp'], "r", encoding="utf-8") as f:
        str_labels = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    chars = ["<unk>"] + sorted([char for char in set(str_labels.lower())])
    char_map = {token: idx for idx, token in enumerate(chars)}
    # lstm前馈
    layer = nn.LSTM(input_size=len(chars), hidden_size=args["num_hiddens"], num_layers=args["num_layers"])
    net = model.RNNModel(layer, vocab_size=len(chars))
    net.load_state_dict(torch.load("checkpoints/model_lstm300.pth"))
    header = "When I had started with the Time Machine, I had started with the absurd assumption that the men of the Future would certainly be infinitely ahead"
    generate(net, args['fp'], header)  # my appliances just as
