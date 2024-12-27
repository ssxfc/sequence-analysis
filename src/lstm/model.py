import torch.nn as nn


class Net(nn.Module):
    """
        :param vocab_size 表示输入单词的格式
        :param embedding_dim 表示将一个单词映射到embedding_dim维度空间
        :param hidden_dim 表示lstm输出隐藏层的维度
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        #Embedding层，将单词映射成vocab_size行embedding_dim列的矩阵，一行的坐标代表第一行的词
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        #两层lstm，输入词向量的维度和隐藏层维度
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, batch_first=False)
        #最后将隐藏层的维度转换为词汇表的维度
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        #获取输入的数据的时间步和批次数
        seq_len, batch_size = input.size()
        #如果没有传入上一个时间的隐藏值，初始一个，注意是2层
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        #将输入的数据embeddings为（input行数,embedding_dim）
        embeds = self.embeddings(input)     # (seq_len, batch_size, embedding_dim), (1,1,128)
        output, hidden = self.lstm(embeds, (h_0, c_0))      #(seq_len, batch_size, hidden_dim), (1,1,256)
        output = self.linear1(output.view(seq_len*batch_size, -1))      # ((seq_len * batch_size),hidden_dim), (1,256) → (1,8293)
        return output, hidden
