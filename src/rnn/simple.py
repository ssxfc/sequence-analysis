import torch
import torch.nn as nn

# 定义输入数据
batch_size = 3   # 批次大小
sequence_length = 6   # 时间步个数
input_size = 10   # 输入特征的维度

# 创建随机输入数据
#输入数据的维度为(sequence_length, batch_size, input_size)，表示有sequence_length个时间步，
#每个时间步有batch_size个样本，每个样本的特征维度为input_size。
input_data = torch.randn(batch_size, sequence_length, input_size)
print("输入数据大小: ",input_data.shape)
# 定义RNN模型
# 定义RNN模型时，我们指定了输入特征的维度input_size、隐藏层的维度hidden_size、隐藏层的层数num_layers等参数。
# batch_first=False表示输入数据的维度中批次大小是否在第一个维度，我们在第二个维度上。
rnn = nn.RNN(input_size, hidden_size=20, num_layers=1, batch_first=True)
"""
在前向传播过程中，我们将输入数据传递给RNN模型，并得到输出张量output和最后一个时间步的隐藏状态hidden。
输出张量的大小为(sequence_length, batch_size, hidden_size)，表示每个时间步的隐藏层输出。
最后一个时间步的隐藏状态的大小为(num_layers, batch_size, hidden_size)。
"""
# 前向传播，第二个参数h0未传递，默认为0
output, final_hidden = rnn(input_data)
print("隐藏层大小: ",output.shape)
print("最后一个隐藏层大小: ",final_hidden.shape)

print(output[:, 5, :] == final_hidden[0])
