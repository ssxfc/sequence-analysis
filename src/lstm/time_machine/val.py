import torch
import torch.utils

from torch import nn


def val(net:nn.Module, val_iter, device):
    net.eval()

    state = None
    loss = nn.CrossEntropyLoss().to(device)
    net = net.to(device)

    total_loss = 0.0
    true_pred = 0
    total = 0
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
        return total_loss, true_pred * 1.0 / total
