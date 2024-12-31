import torch
import torch.utils

from torch import nn

import utils
import dataset
import model


def train(net, data_iter, num_epochs, lr):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    loss = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    loss_list = []
    state = None
    for epoch in range(num_epochs):
        net.train()
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
        loss_list.append(l.detach().cpu().item())
    with open("loss.txt", 'w') as f:
        for l in loss_list:
            f.write(f"{str(l)}\n")
    torch.save(net.state_dict(), "./checkpoints/RNN_Model.pths")


if __name__ == "__main__":
    utils.set_seed(37)
    with open(r"/home/dcd/zww/repos/sequence-analysis/data/timemachine.txt", "r", encoding="utf-8") as f:
        txt_b = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    # config
    args = {
        "dataset": txt_b,
        "num_epochs": 200,
        "batch_size": 2,
        "lr": 0.0001,
        "num_steps": 20,
        "num_hiddens": 4096,
        "num_layers": 1,
    }

    # 加载数据
    data_iter = dataset.SeqDataLoader(args["dataset"], batch_size=args["batch_size"], num_steps=args["num_steps"]) 

    print(" * txt -> idx: ", data_iter.corpus)
    print(" * tok -> idx: ", data_iter.tokens_to_idx)
    print(" * idx -> tok: ", data_iter.idx_to_tokens)

    # lstm前馈
    layer = nn.LSTM(input_size=len(data_iter.tokens_to_idx), hidden_size=args["num_hiddens"], num_layers=args["num_layers"])
    net = model.RNNModel(layer, vocab_size=len(data_iter.tokens_to_idx))

    train(net, data_iter, args["num_epochs"], args["lr"])
