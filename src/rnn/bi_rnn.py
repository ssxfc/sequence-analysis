import torch
import torch.nn as nn

# 定义输入数据
input_size = 10   # 输入特征的维度
sequence_length = 5   # 时间步个数
batch_size = 3   # 批次大小

# 创建随机输入数据
input_data = torch.randn(sequence_length, batch_size, input_size)

# 定义双向RNN模型
rnn = nn.RNN(input_size, hidden_size=20, num_layers=1, batch_first=False, bidirectional=True)

# 前向传播
output, hidden = rnn(input_data)

# 输出结果
print("输出张量大小：", output.size())
print("最后一个时间步的隐藏状态大小：", hidden.size())
