import torch.nn as nn
import torch as t
from poetry import PoetryDataset
from model import Net
num_epochs=5
data_root=r"D:\datasets\poetry.npz"
batch_size=10


def generate(model, start_words, ix2word, word2ix):     # 给定几个词，根据这几个词生成一首完整的诗歌
    txt = []
    for word in start_words:
        txt.append(word)
    input = t.Tensor([word2ix['<START>']]).view(1,1).long()      # tensor([8291.]) → tensor([[8291.]]) → tensor([[8291]])
    input = input.cuda()
    hidden = None
    num = len(txt)
    for i in range(48):      # 最大生成长度
        output, hidden = model(input, hidden)
        if i < num:
            w = txt[i]
            input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0]
            w = ix2word[top_index.item()]
            txt.append(w)
            input = (input.data.new([top_index])).view(1, 1)
        if w == '<EOP>':
            break
    return ''.join(txt)


def train(**kwargs):
    datasets=PoetryDataset(data_root)
    data,ix2word,word2ix=datasets.getData()
    lenData=len(data)
    data = t.from_numpy(data)
    dataloader = t.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=1)
    #总共有8293的词。模型定义：vocab_size, embedding_dim, hidden_dim = 8293 * 128 * 256
    model=Net(len(word2ix),128,256)
    #定义损失函数
    criterion = nn.CrossEntropyLoss()
    model=model.cuda()
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    iteri=0
    filename = "example.txt"
    totalIter=lenData*num_epochs/batch_size
    for epoch in range(num_epochs):  # 最大迭代次数为8
        for i, data in enumerate(dataloader):  # 一批次数据 125*10
            data = data.long().transpose(0,1).contiguous() .cuda()
            optimizer.zero_grad()
            input, target = (data[:-1, :]), (data[1:, :])
            output, _ = model(input)
            loss = criterion(output, target.view(-1))  # torch.Size([15872, 8293]), torch.Size([15872])
            loss.backward()
            optimizer.step()
            iteri+=1
            if(iteri%500==0):
                print(str(iteri+1)+"/"+str(totalIter)+"epoch")
            if (1 + i) % 1000 == 0:  # 每575个batch可视化一次
                with open(filename, "a") as file:
                    file.write(str(i) + ':' + generate(model, '床前明月光', ix2word, word2ix)+"\n")
    t.save(model.state_dict(), './checkpoints/model_poet_2.pth')


def test():
    datasets = PoetryDataset(data_root)
    data, ix2word, word2ix = datasets.getData()
    modle = Net(len(word2ix), 128, 256)  # 模型定义：vocab_size, embedding_dim, hidden_dim —— 8293 * 128 * 256
    if t.cuda.is_available() == True:
        modle.cuda()
        modle.load_state_dict(t.load('./checkpoints/model_poet_2.pth'))
        modle.eval()
        name = input("请输入您的开头：")
        txt = generate(modle, name, ix2word, word2ix)
        print(txt)


if __name__ == "__main__":
    train()
