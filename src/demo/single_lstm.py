import torch
import torch.nn as nn


sequence_length =3
batch_size =2
input_size =4
#这里如果是输入比如[张三,李四，王五]，一般实际使用需要通过embedding后生成一个[时间步是3，批量1（这里是1，但是如果是真实数据集可能有分批处理，就是实际的批次值）,3（三个值的坐标表示一个张三或者李四）]
input=torch.randn(sequence_length,batch_size,input_size)
lstmModel=nn.LSTM(input_size,3,1)
#其中，output是RNN每个时间步的输出，hidden是最后一个时间步的隐藏状态。
output, (h, c) =lstmModel(input)
#因为是3个时间步，每个时间步都有一个隐藏层，每个隐藏层都有2条数据，隐藏层的维度是3，最终(3,2,3)
print("LSTM隐藏层输出的维度",output.shape)
#
print("LSTM隐藏层最后一个时间步输出的维度",h.shape)
print("LSTM隐藏层最后一个时间步细胞状态",c.shape)
