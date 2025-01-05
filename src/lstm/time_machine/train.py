import torch
import random
import torch.utils
from torch import nn
import torchinfo

from tqdm import tqdm

import utils
import dataset
import model
from val import val


def get_params_scope(net):
        # 计算并打印模型参数量
    total_params = 0
    for param in net.parameters():
        total_params += param.numel()
    print(f"模型的总参数量: {total_params}")
    print(torchinfo.summary(net))


def train(net, data_iter, val_iter, device, num_epochs, lr):
    net = net.to(device)
    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200, ], gamma=0.1)
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
            # get_params_scope(net)
            l = loss(y_hat, y.long())
            # loss function正则化, 通过修正权重参数张量，使模型倾向于小权重，从而降低过拟合风险
            # l2_reg = 0.0
            # for param in net.parameters():
            #     l2_reg += torch.norm(param, 2)
            # l += l2_reg * 0.05
            l.backward()
            # 将梯度张量适当缩放，修正其范数
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
            optimizer.step()
            epoch_loss += l.detach().cpu().item()
        scheduler.step()
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
    # 批次数不影响参数量
    # 训练时，时间步可以是任意的，且时间步不影响参数量
    args = {
        "num_epochs": 100,
        "batch_size": 8,
        "lr": 0.0001,
        "num_steps": 64,
        "num_hiddens": 4096,
        "num_layers": 3,
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
    # net.load_state_dict(torch.load("./checkpoints/model_lstm250.pth"))
    train(net, train_data_iter, val_data_iter, device, args["num_epochs"], args["lr"])
