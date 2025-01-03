import torch
import torch.utils
from torch import nn

from tqdm import tqdm

import utils
import dataset
import model
from val import val


def train(net, data_iter, val_iter, device, num_epochs, lr):
    net = net.to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    state = None
    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0.0
        for x, y in tqdm(data_iter, desc=f'epoch:{epoch}/{num_epochs}'):
            x, y = x.to(device), y.to(device)
            y = y.T.reshape(-1)
            # reform the cell and h of previous time step
            if state is None:
                state = net.begin_state(batch_size=x.shape[0], device=device)
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple): 
                    state.detach_()
                else:
                    for s in state:
                        s.detach_()
            optimizer.zero_grad()
            y_hat, state = net(x, state)
            l = loss(y_hat, y.long())
            l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss += l.detach().cpu().item()

        # statistics
        val_loss, acc = val(net, val_iter, device)
        val_loss_list.append(val_loss)
        val_acc_list.append(acc)
        train_loss_list.append(epoch_loss)
        utils.axis_plot("val loss",
                        {"name":"epoch", "list": [i for i in range(epoch + 1)]},
                        {"name":"loss", "list": val_loss_list}
                        )
        utils.axis_plot("train loss",
                        {"name":"epoch", "list": [i for i in range(epoch + 1)]},
                        {"name":"loss", "list": train_loss_list}
                        )
        utils.axis_plot("val acc",
                        {"name":"epoch", "list": [i for i in range(epoch + 1)]},
                        {"name":"loss", "list": val_acc_list}
                        )
        torch.save(net.state_dict(), f"./checkpoints/model_lstm.pth")


if __name__ == "__main__":
    utils.set_seed(37)
    args = {
        "num_epochs": 20,
        "batch_size": 4,
        "lr": 0.0001,
        "num_steps": 20,
        "num_hiddens": 128,
        "num_layers": 1,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(r"/home/dcd/zww/repos/sequence-analysis/data/train.txt", "r", encoding="utf-8") as f:
        train_txt = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    with open(r"/home/dcd/zww/repos/sequence-analysis/data/val.txt", "r", encoding="utf-8") as f:
        val_txt = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    
    # loading data
    train_data_iter = dataset.SeqDataLoader(train_txt, batch_size=args["batch_size"], num_steps=args["num_steps"])
    val_data_iter = dataset.SeqDataLoader(val_txt, batch_size=args["batch_size"], num_steps=args["num_steps"])
    print(" * idx -> tok: ", train_data_iter.idx_to_tokens)
    # setting network
    layer = nn.LSTM(input_size=len(train_data_iter.tokens_to_idx), hidden_size=args["num_hiddens"], num_layers=args["num_layers"])
    net = model.RNNModel(layer, vocab_size=len(train_data_iter.tokens_to_idx))

    train(net, train_data_iter, val_data_iter, device, args["num_epochs"], args["lr"])
