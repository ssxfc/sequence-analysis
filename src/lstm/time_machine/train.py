import torch
import torch.utils
import random
import numpy as np

from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from IPython import display

import utils
import dataset
import model


def train(data_iter, net, num_epochs, lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f" * init train on {device}...")
    print(f" * Current Model {net.rnn._get_name()}...")
    
    net = net.to(device)
    net.train()
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=[0.5, 0.999])
    
    loss_list = []
    state = None
    for epoch in range(num_epochs):
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y = y.T.reshape(-1)
            
            if state is None:
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple): 
                    state.detach_()
                else:
                    for s in state:
                        s.detach_()
                        
            opt.zero_grad()
            y_hat, state = net(X, state)
            l = loss(y_hat, y.long())
            l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            opt.step()
        print(f" * Current Loss: {l.detach().cpu():.4f}, Epoch: {epoch}/{num_epochs}")
        loss_list.apend(l.detach().cpu().item())
    with open("loss.txt", 'w') as f:
        for l in loss_list:
            f.write(l, '\n')
    torch.save(net.state_dict(), "./pths/RNN_Model.pths")


if __name__ == "__main__":
    with open(r"D:\py\engineering\sequence-analysis\.vscode\data\timemachine.txt", "r", encoding="utf-8") as f:
        txt_b = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    """
    对于 CFG_FOR_B:
        可以自由设置。
    """
    CFG_FOR_B = {
        "dataset": txt_b,
        "num_epochs": 1,
        "batch_size": 8,
        "lr": 0.0001,
        "num_steps": 20,
        "num_hiddens": 4096,
        "num_layers": 1,
    }


    utils.set_seed(37)

    # 加载数据
    print(f" * Load Data...")
    args = CFG_FOR_B
    data_iter = dataset.SeqDataLoader(args["dataset"], batch_size=args["batch_size"], num_steps=args["num_steps"]) 

    print(" * txt -> idx: ", data_iter.corpus)
    print(" * tok -> idx: ", data_iter.tokens_to_idx)
    print(" * idx -> tok: ", data_iter.idx_to_tokens)

    # 定义前向层
    # layer = nn.RNN(input_size=len(data_iter.tokens_to_idx), hidden_size=args["num_hiddens"], num_layers=args["num_layers"])
    layer = nn.LSTM(input_size=len(data_iter.tokens_to_idx), hidden_size=args["num_hiddens"], num_layers=args["num_layers"])
    net = model.RNNModel(layer, vocab_size=len(data_iter.tokens_to_idx))

    train(data_iter, net, args["num_epochs"], args["lr"])
