import unittest

import torch
import torch.nn as nn


class TestEmbedding(unittest.TestCase):
    def test_embedding(self):
         # 创建词汇表
        vocab = {"I": 0, "love": 1, "deep": 2, "learning": 3, "!": 4}
        strings=["I", "love", "deep"]
        # 将字符串序列转换为整数索引序列
        input = torch.LongTensor([vocab[word] for word in strings])
        #注意第一个参数是词汇表的个数，并不是输入单词的长度，你在这里就算填100也不影响最终的输出维度，这个输入值影响的是算出来的行向量值
        #nn.Embedding模块会随机初始化嵌入矩阵。在深度学习中，模型参数通常会使用随机初始化的方法来开始训练，以便模型能够在训练过程中学习到合适的参数值。
        #在nn.Embedding中，嵌入矩阵的每个元素都会被随机初始化为一个小的随机值，这些值将作为模型在训练过程中学习的可训练参数，可以使用manual_seed固定。
        torch.manual_seed(1234)
        embedding=nn.Embedding(len(vocab),3)
        print(embedding(input))
