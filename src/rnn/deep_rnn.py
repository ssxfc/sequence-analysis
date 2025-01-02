import torch
import torchinfo
import torch.nn as nn

# 定义输入数据和参数
batch_size = 3
sequence_length = 4

input_size = 5
hidden_size = 10
num_layers = 5

# 创建输入张量
input_tensor = torch.randn(sequence_length, batch_size, input_size)

# 创建多层RNN模型
rnn = nn.RNN(input_size, hidden_size, num_layers)

# 前向传播
output, hidden = rnn(input_tensor)

# 打印输出张量和隐藏状态的大小

print("输入数据大小: ",input_tensor.shape)
print("隐藏层 shape:", output.shape)
print("final Hidden state shape:", hidden[1].shape)
print(torchinfo.summary(rnn))
