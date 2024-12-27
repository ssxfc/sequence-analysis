import torch
import torch.nn as nn


sequence_length =3
batch_size =2
input_size =4
input=torch.randn(sequence_length,batch_size,input_size)
print(input.shape)
lstmModel=nn.LSTM(input_size,3,num_layers=2)
#其中，output是RNN每个时间步的输出，hidden是最后一个时间步的隐藏状态。
output, (h, c) =lstmModel(input)
print("2层LSTM隐藏层输出的维度",output.shape)
print("2层LSTM隐藏层最后一个时间步输出的维度",h.shape)
print("2层LSTM隐藏层最后一个时间步细胞状态",c.shape)
