import os
import random

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def axis_plot(title, x: dict, y: dict, save=True):
    r"""二维坐标图，可用户画dice,loss,precision,recall等随epoch的变化曲线.

    Args:
        title: 二维坐标图名称
        x: x坐标集，字典类型，{'name':'axis name', 'list': [...]}
        y: y坐标集，字典类型，{'name':'axis name', 'list': [...]}
        save: 是否保存图像
    """
    fig, axis = plt.subplots()
    axis.set_title(title)
    axis.set_xlabel(x['name'])
    axis.set_ylabel(y['name'])
    axis.axis([0, max(x['list']), 0, max(y['list'])])
    axis.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.plot(x['list'], y['list'])
    if save:
        plt.savefig(os.path.join('tmp', f'{title.replace(" ", "_")}.png'))
    else:
        plt.show()
    plt.close()

    
def generate_label(fp):
    r"""统计fp文件中不重复字符并对其编号（位置编码）
    Args:
        fp (str): 全量文件路径，也就是train.txt,val.txt,test.txt内容所组成的文件的路径
    """
    with open(fp, "r", encoding="utf-8") as f:
        str_data = "".join(f.readlines()).replace("\n", "").replace("\ufeff", "")
    idx_to_tokens = sorted([char for char in set(str_data.lower())])
    with open("data/labels.txt", "w") as f:
        for char in idx_to_tokens:
            f.write(char + "\n")


if __name__ == "__main__":
    generate_label("/home/dcd/zww/repos/sequence-analysis/data/timemachine.txt")
