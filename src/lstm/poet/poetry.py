from torch.utils.data import  Dataset,DataLoader
import numpy as np


class PoetryDataset(Dataset):
    def __init__(self,root):
        self.data=np.load(root, allow_pickle=True)
    def __len__(self):
        return len(self.data["data"])
    def __getitem__(self, index):
        return self.data["data"][index]
    def getData(self):
        return self.data["data"],self.data["ix2word"].item(),self.data["word2ix"].item()


if __name__=="__main__":
    datas=PoetryDataset(r"D:\datasets\poetry.npz").data
    # data是一个57580 * 125的numpy数组，即总共有57580首诗歌，每首诗歌长度为125个字符（不足125补空格，超过125的丢弃）
    print(datas["data"].shape)
    #这里都字符已经转换成了索引
    print(datas["data"][0])
    # 使用item将numpy转换为字典类型，ix2word存储这下标对应的字,比如{0: '憁', 1: '耀'}
    ix2word = datas['ix2word'].item()
    # print(ix2word)
    # word2ix存储这字对应的小标，比如{'憁': 0, '耀': 1}
    word2ix = datas['word2ix'].item()
    # print(word2ix)
    # 将某一首古诗转换为索引表示,转换后：[5272, 4236, 3286, 6933, 6010, 7066, 774, 4167, 2018, 70, 3951]
    str="床前明月光，疑是地上霜"
    print([word2ix[i] for i in str])
    print([ix2word[i] for i in [word2ix[i] for i in str]])
    #将第一首古诗打印出来
    print([ix2word[i] for i in datas["data"][0]])
