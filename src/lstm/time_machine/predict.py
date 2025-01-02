import torch
import torch.utils

from torch import nn

import utils
import dataset
import model


def infer(net:nn.Module, test_iter):
    r"""预测函数"""
    net.eval()
    state = None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    with torch.no_grad():
        for x, y in test_iter:
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

            # 取第0批数据进行比对
            tmp = y_hat.view(20, 3, -1)
            y_hat = tmp[:, 0, :]
            pred_y = [y_hat.softmax(dim=1).argmax(dim=1)[idx].item() for idx in range(y_hat.size(0))]
            print("".join([test_iter.idx_to_tokens[ii] for ii in pred_y]))
            
            tmp = y.view(20, 3, -1)
            y = tmp[:, 0, :]
            true_y = [y[idx].item() for idx in range(y.size(0))]
            print("".join([test_iter.idx_to_tokens[ii] for ii in true_y]))


if __name__ == "__main__":
    utils.set_seed(37)
    with open(r"D:\py\engineering\sequence-analysis\data\test.txt", "r", encoding="utf-8") as f:
        txt_b = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    args = {
        "dataset": txt_b,
        "num_epochs": 10,
        "batch_size": 3,
        "lr": 0.0001,
        "num_steps": 20,
        "num_hiddens": 4096,
        "num_layers": 2,
    }
    # 加载数据
    test_iter = dataset.SeqDataLoader(args["dataset"], batch_size=args["batch_size"], num_steps=args["num_steps"])
    print(" * idx -> tok: ", test_iter.idx_to_tokens)
    # lstm前馈
    layer = nn.LSTM(input_size=len(test_iter.tokens_to_idx), hidden_size=args["num_hiddens"], num_layers=args["num_layers"])
    net = model.RNNModel(layer, vocab_size=len(test_iter.tokens_to_idx))
    net.load_state_dict(torch.load("checkpoints/RNN_Model.pth"))
    infer(net, test_iter)
