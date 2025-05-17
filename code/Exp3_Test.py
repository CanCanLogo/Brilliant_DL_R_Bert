import torch
import torch.nn as nn
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用的gpu的编号，使用第 0 个

import pickle

from tqdm import tqdm
from torch.utils.data import DataLoader
from Exp3_DataSet import TextDataSet, TestDataSet
from Exp3_Model import Bert_model
from Exp3_Config import Training_Config

def predict(data_loader_test):
    '''
    读取训练好的模型对测试集进行预测，并生成结果文件
    '''
    test_pred = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader_test, dynamic_ncols=True):
            batch_x = data.to(device)

            mask = batch_x['attention_mask'].to(device)
            input_id = batch_x['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)

            y_hat = torch.tensor([torch.argmax(_) for _ in output])

            test_pred += y_hat.tolist()

    # 写入文件
    with open("exp2_predict_labels_1120211392.txt", "w") as f:
        for idx, label_idx in enumerate(test_pred):
            f.write(str(label_idx) + "\n")

if __name__ == '__main__':
    config = Training_Config()

    test_dataset = TestDataSet(filepath="../data/test_exp3.txt", max_token_per_sent=config.max_token_per_sent)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.test_batch_size)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = torch.load("model_bert.pth").to(device)
    # 对测试集进行预测
    predict(test_loader)
