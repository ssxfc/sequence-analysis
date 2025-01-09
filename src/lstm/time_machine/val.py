import torch
import torch.utils

from torch import nn

import model
import dataset


def val(net:nn.Module, val_iter, device):
    net.eval()

    state = None
    loss = nn.CrossEntropyLoss().to(device)
    net = net.to(device)

    total_loss = 0.0
    true_pred = 0
    total = 0
    num_batch = 0
    with torch.no_grad():
        for x, y in val_iter:
            x, y = x.to(device), y.to(device)
            y = y.T.reshape(-1)
            if state is None:
                state = net.begin_state(batch_size=x.shape[0], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple): 
                    state.detach_()
                else:
                    for s in state:
                        s.detach_()
            y_hat, state = net(x, state)
            l = loss(y_hat, y.long())

            # 采集数据
            pred_y = y_hat.softmax(dim=1).argmax(dim=1)
            total_loss += l.detach().cpu().item()
            true_pred += (pred_y == y).sum().item()
            total += y.size(0)
            num_batch += 1
        return total_loss / num_batch, true_pred * 1.0 / total


if __name__ == "__main__":
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

    with open(r"/home/dcd/zww/repos/sequence-analysis/data/train.txt", "r", encoding="utf-8") as f:
        train_txt = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    # loading data
    train_data_iter = dataset.SeqDataLoader(train_txt, batch_size=args["batch_size"], num_steps=args["num_steps"])
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # lstm前馈
    layer = nn.LSTM(input_size=len(chars), hidden_size=args["num_hiddens"], num_layers=args["num_layers"])
    net = model.RNNModel(layer, vocab_size=len(chars))
    net.load_state_dict(torch.load("checkpoints/model_lstm300.pth"))
    loss, acc = val(net, train_data_iter, device)
    print(loss)
    print(acc)
