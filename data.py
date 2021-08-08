import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def read_corpus(corpus_path):
    """
    输入文本为(word+label)行文本"w1 l1\n w2 l2\n..."，输出为[sent1,sent2,..,], sent=([w1,w2,...],[l1,l2,...])
    read corpus and return the list of samples
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()  # 得到一个列表，每一个元素代表一行的字符串
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':  # line不是句尾
            [char, label] = line.strip().split()
            sent_.append(char)  # 存放该句的字符序列
            tag_.append(label)  # 存放该句的标签序列
        else:
            data.append((sent_, tag_)) # 添加一句的信息
            sent_, tag_ = [], []

    return data

def vocab_build(vocab_path, corpus_path, min_count):
    """
    将语料库中的字符进行计数统计，并形成字典，去掉低频字后，每种字符对应于一个编号。
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """

    corpus_path = "data_path/train_data"
    vocab_path = "test_voca.pkl"
    min_count = 5

    data = read_corpus(corpus_path)
    word2id = {}  # 词典
    for sent_, tag_ in data:
        for word in sent_:  # 一句话中的一个字
            if word.isdigit():
                word = '<NUM>'  # 数字
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):  # 为什么要如此表达英文字母
                word = '<ENG>'
            if word not in word2id:  # 该类型字符不存在的话
                word2id[word] = [len(word2id)+1, 1]  # 遇到第一个数字的话，。word2id={"NUM":[1,1]}
            else:
                word2id[word][1] += 1  # 遇到第二个数字,word2id={"NUM":[1,2]}，其中的列表第一个元素代表类型的索引，第二个代表该类型字符的数量

    low_freq_words = []  # 低频字列表
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':  # 字符频次小于阈值并且不属于字母和数字，才算是低频字
            low_freq_words.append(word)
    for word in low_freq_words:  # 从词典中删除低频字
        del word2id[word]

    new_id = 1
    for word in word2id.keys():  # 重新给词汇类型编号
        word2id[word] = new_id  # ？？？？？这儿不是覆盖了之前的计数吗？还是说词典根本不需要计数，计数只是为了剔除低频字。
        new_id += 1
    word2id['<UNK>'] = new_id  # 给未知词编号
    word2id['<PAD>'] = 0  # 给填充物编号

    print(len(word2id))  # 词典大小
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """
    把一句话按照字典里的索引转化成编号的序列
    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():  # 判断字符是不是数字
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):  # 判断字符是不是英文字母
            word = '<ENG>'
        if word not in word2id:  # 判断字符是不是未知字
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id


def read_dictionary(vocab_path):
    """
`   读取指定路径的字典为dict对象
    :param vocab_path:
    :return:
    """
    vocab_path = os.path.join(vocab_path)  # 这样操作是为了将路径规范为双斜杠吗？好像不太像。
    with open(vocab_path, 'rb') as fr: # 以二进制方式读取字典
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))  # 构建一个均匀分布的随机嵌入矩阵，将长为|V|的向量变换为指定嵌入维度
    embedding_mat = np.float32(embedding_mat)  # 这一步应该可以合并到上一步中
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    """
    输入序列的列表，将序列进行右填充指定填充符至同等长度，此处的序列应该语句转化成字符编号的序列，填充符0则对应着词典中的UNK
    :param sequences: [sent1,sent2,...],  sent = ["我","爱","你"]
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))  # 得到句子的序列中的最长句子的长度
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)  # 使用指定填充符将句子填充到指定长度，右填充
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab: 词典
    :param tag2label: 将标签转化为编号的字典
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data)  # 混淆样本，一个样本就是一句话及其对应标签

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)  # 将句子转换为ID序列
        label_ = [tag2label[tag] for tag in tag_]  # 将标签也转化为编号

        if len(seqs) == batch_size:  # 每次seqs收集到batch_size个样本则生成该批次的数据和标签
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:  # 最后输出不足batch个的剩余样本
        yield seqs, labels

