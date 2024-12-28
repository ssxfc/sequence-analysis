import torch
import torch.utils
import random
import numpy as np


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)
    
    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
        

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序采样生成一个小批量子序列"""
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = np.array(corpus[offset : offset + num_tokens])
    Ys = np.array(corpus[offset + 1 : offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        yield torch.tensor(X), torch.tensor(Y)


def get_train_data(txt):
    idx_to_tokens = ["<unk>"] + sorted([char for char in set(txt.lower())])
    tokens_to_idx = {token: idx for idx, token in enumerate(idx_to_tokens)}
    corpus = [tokens_to_idx[token] for token in txt.lower()]
    return corpus, tokens_to_idx, idx_to_tokens


class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, txt, batch_size, num_steps):
        self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.tokens_to_idx, self.idx_to_tokens = get_train_data(txt) 
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
