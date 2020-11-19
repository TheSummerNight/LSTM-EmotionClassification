import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

import config
from utils import tokenize

ws = pickle.load(open("./model/ws.pkl", "rb"))


class ImdbDataset(Dataset):
    def __init__(self, train=True):
        super(ImdbDataset, self).__init__()
        data_path = r"data/aclImdb"
        data_path += r"/train" if train else "/test"
        self.total_path = []
        for temp_path in [r"\pos", r"\neg"]:
            cur_path = data_path + temp_path
            # 添加积极和消极评论的所有文件
            self.total_path += [os.path.join(cur_path, i) for i in os.listdir(cur_path) if i.endswith(".txt")]

    def __getitem__(self, idx):
        # 获取某评论的文件路径
        file = self.total_path[idx]
        # 读取评论内容
        content = open(file=file, encoding='utf-8').read()
        # 将评论分成一个个单词列表
        content = tokenize(content)
        # 获取评论的分数（小于5为消极，大于等于5为积极）
        score = int(file.split("_")[1].split(".")[0])
        label = 0 if score < 5 else 1
        return content, label

    def __len__(self):
        return len(self.total_path)


def collate_fn(batch):
    """
    对batch数据进行处理([tokens,label],[tokens,label]...)
    :param batch:
    :return:
    """
    # *batch 可理解为解压，返回二维矩阵式
    content, labels = list(zip(*batch))
    # content中是有batch_size个评论（句子）
    content = [ws.transform(sentence, 200) for sentence in content]
    # content式字符串数组，必须先将数组中字符转化成对应数字，才能转成张量
    content = torch.LongTensor(content)
    labels = torch.LongTensor(labels)

    return content, labels


def get_dataloader(train=True):
    imdb_dataset = ImdbDataset(train)
    return DataLoader(imdb_dataset,
                      batch_size=config.train_batch_size if train else config.test_batch_size,
                      shuffle=True,
                      collate_fn=collate_fn)


if __name__ == '__main__':
    # dataset = ImdbDataset(True)
    # print(dataset[0])
    # print(len(get_dataloader(True)))
    # print(type(get_dataloader()))
    # exit()
    # print(get_dataloader())
    # exit()
    for idx, (content, label) in enumerate(get_dataloader(True)):
        # for i,con in enumerate(content,0):
        #     print(i,con)
        print(idx)
        print(content)
        print(label)
        break
