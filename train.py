"""
训练模型
"""
import torch
import config
from word_sequence import WordSequence
from lstm_model import LSTM_Model
from dataset import get_dataloader
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = config.device

loss_list = []


def train(epoch):
    model = LSTM_Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_dataloader = get_dataloader(train=True)
    bar = tqdm(train_dataloader, total=len(train_dataloader))
    for idx, (input, target) in enumerate(bar):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        loss_list.append(loss.cpu().data)
        optimizer.step()
        bar.set_description("epoch:{} idx:{} loss:{:.6f}".format(epoch, idx, np.mean(loss_list)))
        if idx % 10 == 0:
            torch.save(model.state_dict(), 'model/model.pkl')
            torch.save(optimizer.state_dict(), 'model/optimizer.pkl')


def eval():
    model = LSTM_Model().to(device)
    model.load_state_dict(torch.load('model/model.pkl'))
    model.eval()
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False)
    with torch.no_grad():
        for input, target in test_dataloader:
            input = input.to(config.device)
            target = target.to(config.device)
            output = model(input)
            loss = F.nll_loss(output, target)
            loss_list.append(loss.item())
            # 准确率
            output_max = output.max(dim=-1)
            pred = output_max[-1]
            acc_list.append(pred.eq(target).cpu().float().mean())
        print("loss:{:.6f},acc:{}".format(np.mean(loss_list), np.mean(acc_list)))


if __name__ == '__main__':
    for i in range(10):
        train(i)
        eval()
    plt.figure(figsize=(20, 8))
    plt.plot(range(len(loss_list)), loss_list)
