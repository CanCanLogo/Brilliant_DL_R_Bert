import torch
import torch.nn as nn
import time
import json
import os

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"	#使用的gpu的编号，使用第 0 个

import pickle

from tqdm import tqdm
from torch.utils.data import DataLoader
from Exp3_DataSet import TextDataSet
from Exp3_Model import Bert_model
from Exp3_Config import Training_Config


def train(num_epochs, batch_size, data_loader_train, data_loader_valid):
    '''
    进行训练
    '''
    max_valid_acc = 0
    
    for epoch in range(num_epochs):
        model.train()

        total_loss = []
        total_true = []

        tqdm_iterator = tqdm(data_loader_train, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for data in tqdm_iterator:
            # print(data)
            # 选取对应批次数据的输入和标签
            batch_x, batch_y = data[0].to(device), data[1].to(device)

            # print(batch_y)
            # print(batch_x)
            # print(batch_x.shape)
            # 模型预测
            # y_hat = model(batch_x)
            mask = batch_x['attention_mask'].to(device)
            input_id = batch_x['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # print(y_hat)
            batch_y = batch_y.long()

            loss = loss_function(output, batch_y)

            optimizer.zero_grad()   # 梯度清零
            loss.backward()         # 计算梯度
            optimizer.step()        # 更新参数

            y_hat = torch.tensor([torch.argmax(_) for _ in output]).to(device)
            # print(y_hat)
            # print()
            total_true.append(torch.sum(y_hat == batch_y).item())
            total_loss.append(loss.item())

            tqdm_iterator.set_postfix(loss=sum(total_loss) / len(total_loss),
                                      acc=sum(total_true) / (batch_size * len(total_true)))
        
        tqdm_iterator.close()

        train_loss = sum(total_loss) / len(total_loss)
        train_acc = sum(total_true) / (batch_size * len(total_true))

        valid_acc = valid(data_loader_valid, batch_size)

        if valid_acc > max_valid_acc:
            # torch.save(model, os.path.join(output_folder, "model"))
            torch.save(model, "model_bert.pth")
        print(f"epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc*100:.2f}%, valid accuracy: {valid_acc*100:.2f}%")
    # torch.save(model, "model_1" + ".pth")


def valid(data_loader_valid, batch_size):
    '''
    进行验证，返回模型在验证集上的 accuracy
    '''
    total_true = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_valid, dynamic_ncols=True):
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            batch_y = batch_y.long()
            mask = batch_x['attention_mask'].to(device)
            input_id = batch_x['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            # 取分类概率最大的类别作为预测的类别
            y_hat = torch.tensor([torch.argmax(_) for _ in output]).to(device)
            total_true.append(torch.sum(y_hat == batch_y).item())

        return sum(total_true) / (batch_size * len(total_true))

if __name__ == '__main__':
    config = Training_Config()

    with open('../data/rel2id.json', 'r', encoding='utf-8') as file:
        dict = json.load(file)

    # 训练集验证集
    train_dataset = TextDataSet(filepath="../data/data_train.txt", id2rel_dict = dict, max_token_per_sent = config.max_token_per_sent)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size)

    val_dataset = TextDataSet(filepath="../data/data_val.txt", id2rel_dict = dict, max_token_per_sent = config.max_token_per_sent)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Bert_model(ntoken=config.max_token_per_sent,
                       d_emb=config.embedding_dimension,
                       num_classes=config.label_num).to(device)
    
    # 设置损失函数
    loss_function = nn.CrossEntropyLoss()
    # 设置优化器                                       
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)

    # 进行训练
    train(config.epoch, config.batch_size, train_loader, val_loader)
