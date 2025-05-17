import torch
import json
import jieba
from torch.utils.data import Dataset
import numpy as np

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(r"D:\new_program\pythonProject\pytorchUse\NLPpro1\NLPpro2\bert-base-chinese")


# 训练集和验证集
class TextDataSet(Dataset):
    def __init__(self, filepath, id2rel_dict, max_token_per_sent):
        self.max_token_per_sent = max_token_per_sent
        self.id2rel = id2rel_dict[0]
        self.rel2id = id2rel_dict[1]
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        self.data = []
        self.label = []
        for line in lines:
            tmp = {}
            tmp2 = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['relation'] = line[2]
            tmp['text'] = line[3][:-1]
            # tmp2['data'] = jieba.lcut(tmp['head'] + '\n'+ tmp['tail'] + '\n' + tmp['text'], cut_all=True)
            tmp2['data'] = '两个实体：' + tmp['head'] + '\n' + tmp['tail'] + '\n' + '分类关系：' + tmp['text']
            tmp2['label'] = self.rel2id[tmp['relation']]
            # print(len(tmp2['data']))
            self.original_data.append(tmp)
            self.data.append(tmp2['data'])
            self.label.append(tmp2['label'])
        self.data = [tokenizer(text,
                          padding='max_length',
                          max_length=self.max_token_per_sent,
                          truncation=True,  # 所有句子都被截断或填充到相同的长度
                          return_tensors="pt")  # 返回PyTorch张量
                for text in self.data]
        print(self.data[0])

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)

# 测试集是没有标签的，因此函数会略有不同
class TestDataSet(Dataset):
    def __init__(self, filepath, max_token_per_sent):
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.max_token_per_sent = max_token_per_sent
        self.original_data = []
        for line in lines:
            tmp = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['text'] = line[2][:-1]
            # sent = jieba.lcut(tmp['head'] + '\n' + tmp['tail'] + '\n' + tmp['text'], cut_all=True)
            sent = '两个实体：' + tmp['head'] + '\n' + tmp['tail'] + '\n' + '分类关系：' + tmp['text']
            self.original_data.append(sent)
        self.original_data = [tokenizer(text,
                               padding='max_length',
                               max_length=self.max_token_per_sent,
                               truncation=True,  # 所有句子都被截断或填充到相同的长度
                               return_tensors="pt")  # 返回PyTorch张量
                     for text in self.original_data]

    def __getitem__(self, index):
        return self.original_data[index]

    def __len__(self):
        return len(self.original_data)

class TextDataSet_new(Dataset):
    def __init__(self, filepath, id2rel_dict, max_token_per_sent):
        self.max_token_per_sent = max_token_per_sent
        self.id2rel = id2rel_dict[0]
        self.rel2id = id2rel_dict[1]
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        self.data = []
        self.label = []
        for line in lines:
            tmp = {}
            tmp2 = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['relation'] = line[2]
            tmp['text'] = line[3][:-1]

            if tmp['tail'] == 'object_placeholder' or tmp['head'] == 'subject_placeholder':
                # print('有两个实体：' + tmp['head'] + '、' + tmp['tail'] + '，分类关系：' + tmp['text'])
                continue

            # tmp2['data'] = jieba.lcut(tmp['head'] + '\n'+ tmp['tail'] + '\n' + tmp['text'], cut_all=True)
            # tmp2['data'] = tmp['head'] + '[SEP]' + tmp['tail'] + '[SEP]' + tmp['text']
            # tmp2['data'] = tmp['text']
            # tmp2['data'] = self.extract_entity(tmp['head'], tmp['tail'], tmp['text'])  # 这里是用来pad的

            tmp2['data'] = '有两个实体：' + tmp['head'] + '、' + tmp['tail'] + '，分类关系：' + tmp['text']

            tmp2['label'] = self.rel2id[tmp['relation']]
            # print(len(tmp2['data']))
            self.original_data.append(tmp)

            encoded = tokenizer(tmp2['data'],
                                max_length=self.max_token_per_sent,
                                return_tensors="pt",
                                padding='max_length',
                                truncation=True)
            input_ids = encoded['input_ids'][0]
            attention_mask = encoded['attention_mask']

            # print(tmp2['data'])

            '''
            以下获得e1_mask
            '''
            # e1_start, e1_end, e2_start, e2_end = self.locate_entity(encoded['input_ids'][0])

            e1_tokens = tokenizer.tokenize(tmp['head'])
            e1_ids = tokenizer.convert_tokens_to_ids(e1_tokens)

            e1_start_idx = None

            e1_mask = torch.zeros_like(attention_mask)
            e1_mask_all = torch.zeros_like(attention_mask)

            sign_1 = 2
            for i in range(len(input_ids) - len(e1_ids) + 1):
                if sign_1 > 0:
                    pass
                else:
                    break
                if input_ids[i:i + len(e1_ids)].tolist() == e1_ids:
                    sign_1 -= 1
                    e1_start_idx = i
                    e1_end_idx = e1_start_idx + len(e1_ids) - 1
                    e1_mask_all[:, e1_start_idx:e1_end_idx + 1] = 1
                    if sign_1 == 0:
                        e1_start_idx = i
                        e1_end_idx = e1_start_idx + len(e1_ids) - 1
                        e1_mask[:, e1_start_idx:e1_end_idx + 1] = 1


            e2_tokens = tokenizer.tokenize(tmp['tail'])
            e2_ids = tokenizer.convert_tokens_to_ids(e2_tokens)

            e2_start_idx = None

            e2_mask = torch.zeros_like(attention_mask)

            sign_2 = 2

            for i in range(len(input_ids) - len(e2_ids) + 1):
                if e1_mask_all[0, i] == 1:
                    continue
                if sign_2 > 0:
                    pass
                else:
                    break
                if input_ids[i:i + len(e2_ids)].tolist() == e2_ids:
                    sign_2 -= 1
                    if sign_2 == 0:
                        e2_start_idx = i
                        e2_end_idx = e2_start_idx + len(e2_ids) - 1
                        e2_mask[:, e2_start_idx:e2_end_idx + 1] = 1

            if sign_1 > 0 or sign_2 > 0:
                print(tmp['head'])
                print(tmp['tail'])
                print(tmp['relation'])
                print(tmp2['data'])
                continue



            # if e1_end_idx > e2_end_idx:
            #     print(tmp2['data'])
            # if e1_start_idx < 20:
            #     print(tmp2['data'])
            # if tmp['tail'] in tmp['head'] or tmp['head'] in tmp['tail']:
            #     print(tmp2['data'])
            # if tmp['head'] in tmp['tail']:
            #     print(e1_start_idx, e1_end_idx, e2_start_idx, e2_end_idx)
            #     print(tmp2['data'])

            # if e1_start_idx is not None:
            #     e1_end_idx = e1_start_idx + len(e1_ids) - 1
            # else:
            #     print(input_ids[e1_start+1:e1_end])
            #     print(e1_ids)
            #     print(tokenizer.convert_ids_to_tokens(input_ids[e1_start+1:e1_end]))
            #     print(tokenizer.convert_ids_to_tokens(e1_ids))
            #
            #     # if tmp['head'] == 'subject_placeholder':
            #     #     # print(tmp['relation'])
            #     #     continue
            #     # # else:
            #     # #     continue
            #     print("Phrase not found in the tokenized sentence.")
            #     print(tmp['head'])
            #     print(tmp['tail'])
            #     print(tmp['relation'])
            #     print(tmp2['data'])
            #     # print(input_ids.tolist())
            #     # try:
            #     #     phrase_start_idx = input_ids.tolist().index(e1_ids[0])
            #     #     phrase_end_index = phrase_start_idx + len(e1_ids) - 1
            #     #     print(input_ids[phrase_start_idx:phrase_end_index].tolist())
            #     #     print(e1_ids)
            #     # except:
            #     #     pass
            #     # 在这里处理短语未找到的情况
            #
            # # 根据找到的token位置，创建一个mask tensor
            # e1_mask = torch.zeros_like(attention_mask)
            # if e1_start_idx is not None:
            #     e1_mask[:, e1_start_idx:e1_end_idx + 1] = 1

            '''
            以下获得e2_mask
            '''

            # if tmp['tail'] in tmp['head']:


            # e2_tokens = tokenizer.tokenize(tmp['tail'])
            # e2_ids = tokenizer.convert_tokens_to_ids(e2_tokens)
            #
            # e2_start_idx = None
            #
            # for i in range(len(input_ids) - len(e2_ids) + 1):
            #     if i >= e1_start_idx and i <= e1_end_idx:
            #         continue
            #     if input_ids[i:i + len(e2_ids)].tolist() == e2_ids:
            #         e2_start_idx = i
            #         break
            #
            # if e2_start_idx is not None:
            #     e2_end_idx = e2_start_idx + len(e2_ids) - 1
            # else:
            #
            #     print(input_ids[e2_start+1:e2_end])
            #     print(e2_ids)
            #     # if tmp['tail'] == 'object_placeholder':
            #     #     # print(print(tmp['relation']))
            #     #     continue
            #     # # else:
            #     # #     continue
            #     print("Phrase not found in the tokenized sentence.")
            #
            #     print(tmp['head'])
            #     print(tmp['tail'])
            #     print(tmp['relation'])
            #     print(tmp2['data'])
            #     # print(input_ids.tolist())
            #     # try:
            #     #     phrase_start_idx = input_ids.tolist().index(e2_ids[0])
            #     #     phrase_end_index = phrase_start_idx + len(e2_ids) - 1
            #     #     print(input_ids[phrase_start_idx:phrase_end_index].tolist())
            #     #     print(e2_ids)
            #     # except:
            #     #     pass
            #     # 在这里处理短语未找到的情况
            #
            # # 根据找到的token位置，创建一个mask tensor
            # e2_mask = torch.zeros_like(attention_mask)
            # if e2_start_idx is not None:
            #     e2_mask[:, e2_start_idx:e2_end_idx + 1] = 1

            encoded['e1_mask'] = e1_mask
            encoded['e2_mask'] = e2_mask
            del encoded['token_type_ids']
            # print(encoded['e1_mask'])
            # print(encoded['e2_mask'])
            self.data.append(encoded)
            self.label.append(tmp2['label'])
            # print(encoded['token_type_ids'])
        # self.data = [tokenizer(text,
        #                   padding='max_length',
        #                   max_length=self.max_token_per_sent,
        #                   truncation=True,  # 所有句子都被截断或填充到相同的长度
        #                   return_tensors="pt")  # 返回PyTorch张量
        #         for text in self.data]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)

    def pad_substring(self, main_string, substring, pad_token="[PAD]"):
        start_idx = main_string.find(substring)
        if start_idx != -1:  # 如果找到了子串
            # 计算结束索引（不包含子串后面的字符）
            end_idx = start_idx + len(substring)
            # 在子串前后添加[PAD]
            padded_substring = pad_token + substring + pad_token
            # 使用字符串切片和连接来构建新的字符串
            new_string = main_string[:start_idx] + padded_substring + main_string[end_idx:]
            return new_string, start_idx, end_idx
        else:
            print('find no entity')
            print(main_string)
            print(substring)
            return main_string, 0, 0

    def pad_substring_2(self, main_string, substring, start, end, pad_token="[PAD]"):
        while 1:
            start_idx = main_string.find(substring)
            if start_idx <= end and start_idx >= start:
                continue
            elif start_idx != -1:  # 如果找到了子串
                # 计算结束索引（不包含子串后面的字符）
                end_idx = start_idx + len(substring)
                # 在子串前后添加[PAD]
                padded_substring = pad_token + substring + pad_token
                # 使用字符串切片和连接来构建新的字符串
                new_string = main_string[:start_idx] + padded_substring + main_string[end_idx:]
                return new_string
            else:
                print('find no entity')
                print(main_string)
                print(substring)
                return main_string

    def extract_entity(self, head, tail, text):
        new_text_1, start, end = self.pad_substring(text, head)

        new_text_2 = self.pad_substring_2(new_text_1, tail, start, end)
        return new_text_2

    def locate_entity(self, nums):
        # for i, num in enumerate(nums):
            # if num != 0:
            #     current_slice.append(num)
            # elif current_slice:
            #     # 检查前一个元素是否也是0（确保切片前面是0）
            #     if i > 0 and nums[i - 1] == 0:
            #         # 将完整的切片添加到切片列表中
            #         slices_found.append(current_slice)
            #         # 重置当前切片为空，准备寻找下一个切片
            #         current_slice = []
            #         # 如果当前切片为空并且遇到0，则忽略（除非它是列表的最后一个元素）
            # elif i < len(nums) - 1 and nums[i + 1] != 0:
            #     continue
            #     # 如果列表在这里结束，并且current_slice非空，则添加它（作为最后一个切片）
            # elif i == len(nums) - 1 and current_slice:
            #     slices_found.append(current_slice)
        arr = np.array(nums)
        zero_indices = np.where(arr == 0)[0]
        return zero_indices[0], zero_indices[1], zero_indices[2], zero_indices[3]



class TestDataSet_new(Dataset):
    def __init__(self, filepath, id2rel_dict, max_token_per_sent):
        self.max_token_per_sent = max_token_per_sent
        self.id2rel = id2rel_dict[0]
        self.rel2id = id2rel_dict[1]
        lines = open(filepath, 'r', encoding='utf-8').readlines()
        self.original_data = []
        self.data = []
        self.label = []
        for line in lines:
            tmp = {}
            tmp2 = {}
            line = line.split('\t')
            tmp['head'] = line[0]
            tmp['tail'] = line[1]
            tmp['text'] = line[2][:-1]
            # tmp2['data'] = jieba.lcut(tmp['head'] + '\n'+ tmp['tail'] + '\n' + tmp['text'], cut_all=True)
            # tmp2['data'] = tmp['head'] + '[SEP]' + tmp['tail'] + '[SEP]' + tmp['text']
            tmp2['data'] = tmp['text']

            # print(len(tmp2['data']))
            self.original_data.append(tmp)

            encoded = tokenizer(tmp2['data'],
                                max_length=self.max_token_per_sent,
                                return_tensors="pt",
                                padding='max_length',
                                truncation=True)
            input_ids = encoded['input_ids'][0]
            attention_mask = encoded['attention_mask']

            '''
            以下获得e1_mask
            '''

            e1_tokens = tokenizer.tokenize(tmp['head'])
            e1_ids = tokenizer.convert_tokens_to_ids(e1_tokens)

            e1_start_idx = None

            for i in range(len(input_ids) - len(e1_ids) + 1):
                if input_ids[i:i + len(e1_ids)].tolist() == e1_ids:
                    e1_start_idx = i
                    break

            if e1_start_idx is not None:
                e1_end_idx = e1_start_idx + len(e1_ids) - 1
            else:
                print("Phrase not found in the tokenized sentence.")
                print(tmp2['data'])
                # 在这里处理短语未找到的情况

            # 根据找到的token位置，创建一个mask tensor
            e1_mask = torch.zeros_like(attention_mask)
            if e1_start_idx is not None:
                e1_mask[:, e1_start_idx:e1_end_idx + 1] = 1

            '''
            以下获得e2_mask
            '''
            e2_tokens = tokenizer.tokenize(tmp['tail'])
            e2_ids = tokenizer.convert_tokens_to_ids(e2_tokens)

            e2_start_idx = None

            for i in range(len(input_ids) - len(e2_ids) + 1):
                if input_ids[i:i + len(e2_ids)].tolist() == e2_ids:
                    e2_start_idx = i
                    break

            if e2_start_idx is not None:
                e2_end_idx = e2_start_idx + len(e2_ids) - 1
            else:
                print("Phrase not found in the tokenized sentence.")
                print(tmp2['data'])
                # 在这里处理短语未找到的情况

            # 根据找到的token位置，创建一个mask tensor
            e2_mask = torch.zeros_like(attention_mask)
            if e2_start_idx is not None:
                e2_mask[:, e2_start_idx:e2_end_idx + 1] = 1

            encoded['e1_mask'] = e1_mask
            encoded['e2_mask'] = e2_mask
            del encoded['token_type_ids']
            # print(encoded['e1_mask'])
            # print(encoded['e2_mask'])
            self.data.append(encoded)

            # print(encoded['token_type_ids'])
        # self.data = [tokenizer(text,
        #                   padding='max_length',
        #                   max_length=self.max_token_per_sent,
        #                   truncation=True,  # 所有句子都被截断或填充到相同的长度
        #                   return_tensors="pt")  # 返回PyTorch张量
        #         for text in self.data]

    def __getitem__(self, index):
        data = self.data[index]
        return data

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    with open('../data/rel2id.json', 'r', encoding='utf-8') as file:
        dict = json.load(file)

    trainset = TextDataSet_new(filepath="../data/data_train.txt", id2rel_dict = dict, max_token_per_sent=280)
    # validset = TextDataSet(filepath="./data/data_val.txt")
    # testset = TestDataSet(filepath="./data/test_exp3.txt")
    print("训练集长度为：", len(trainset))  # 删除了holder部分：36648 原本的长度：37965  删掉了不能找到两个的情况：训练集长度为： 36468
    # print("测试集长度为：", len(testset))
