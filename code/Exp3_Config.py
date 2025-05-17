"""
该文件旨在配置训练过程中的各种参数
请按照自己的需求进行添加或者删除相应属性
"""


class Training_Config(object):
    def __init__(self,
                 embedding_dimension=768,
                 training_epoch=5,
                 max_sentence_length=200,
                 cuda=True,
                 label_num=44,
                 learning_rate=5e-6,
                 batch_size=8,
                 test_batch_size = 16,
                 weight_decay=5e-4
                 dropout=0.1):
        self.embedding_dimension = embedding_dimension  # 词向量的维度
        self.epoch = training_epoch  # 训练轮数
        self.max_sentence_length = max_sentence_length  # 句子最大长度
        self.label_num = label_num  # 分类标签个数
        self.lr = learning_rate  # 学习率
        self.batch_size = batch_size  # 批大小
        self.cuda = cuda  # 是否用CUDA
        self.dropout = dropout  # dropout概率
        self.weight_decay = weight_decay
        self.test_batch_size = test_batch_size

# new:    embedding_dim = 768     # 每个词向量的维度
#     max_token_per_sent = 280  # 每个句子预设的最大 token 数
#     batch_size = 6
#     num_epochs = 3
#     lr = 8e-6

    # embedding_dim = 768     # 每个词向量的维度
    # max_token_per_sent = 200  # 每个句子预设的最大 token 数
    # batch_size = 8
    # num_epochs = 3
    # lr = 5e-6