import torch.nn as nn
import torch as torch
import math
from transformers import BertModel

class Bert_model(nn.Module):
    def __init__(self, d_emb=768, num_classes = 44):
        super(Bert_model, self).__init__()
        self.bert = BertModel.from_pretrained(r"D:\new_program\pythonProject\pytorchUse\NLPpro1\NLPpro2\bert-base-chinese")
        self.fc = nn.Linear(d_emb, num_classes)
    def forward(self ,input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id,
                                     attention_mask=mask,
                                     return_dict=False)
        x = self.fc(pooled_output)
        return x

class Bert_changed(nn.Module):

    def __init__(self, ntoken, d_emb=768, num_classes = 44, dropout=0.1):
        super(Bert_changed, self).__init__()

        self.pretrained_model_path = r"D:\new_program\pythonProject\pytorchUse\NLPpro1\NLPpro2\bert-base-chinese"
        self.embedding_dim = d_emb
        self.dropout = dropout
        self.tagset_size = num_classes

        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.drop = nn.Dropout(self.dropout)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(self.embedding_dim * 3)
        self.hidden2tag = nn.Linear(self.embedding_dim * 3, self.tagset_size)

    def forward(self, token_ids, attention_mask, e1_mask, e2_mask):
        sequence_output, pooled_output = self.bert_model(input_ids=token_ids,
                                                         attention_mask=attention_mask, return_dict=False)

        # 每个实体的所有token向量的平均值
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        e1_h = self.activation(self.dense(e1_h))
        e2_h = self.activation(self.dense(e2_h))

        # [cls] + 实体1 + 实体2
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2tag(self.drop(concat_h))

        return logits

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, max_len, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        # (batch_size,1,max_len)
        e_mask_unsqueeze = e_mask.unsqueeze(1)

        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(
            1)  # [b, 1, max_len] * [b, max_len, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
