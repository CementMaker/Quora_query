import re
import os
import pickle
import random
import statistics

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from collections import Counter
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from extral_features import *

path = os.getcwd()


columns=['question1', 'question2', 'lcs_test', 'edit_distance_test',
         'sentiment1_test', 'sentiment2_test', 'sentiment3_test',
         'sentiment4_test', 'sentiment5_test', 'sentiment6_test',
         'ratio1_test', 'ratio2_test', 'ratio3_test', 'ratio4_test','ratio5_test', 'ratio6_test',
         'length_difference1_test', 'length_difference2_test',
         'tf_idf_word_match_test',
         'is_duplicate']

columns_extra_feature = ['lcs_test', 'edit_distance_test',
                         'sentiment1_test', 'sentiment2_test', 'sentiment3_test',
                         'sentiment4_test', 'sentiment5_test', 'sentiment6_test',
                         'ratio1_test', 'ratio2_test', 'ratio3_test', 'ratio4_test','ratio5_test', 'ratio6_test',
                         'length_difference1_test', 'length_difference2_test',
                         'tf_idf_word_match_test']

def preprocess_tocsv(file_path):
    df_old = pd.read_csv(file_path).dropna(axis=0)
    query_a = [data.text_to_wordlist(text) for text in df_old['question1'].values][0:500000]
    query_b = [data.text_to_wordlist(text) for text in df_old['question2'].values][0:500000]

    query, words = [], []
    for text in np.append(query_a, query_b):
        query.append(text)
        if len(text) < 5:
            print(text)

    length = []
    for text in query:
        length.append(len(text))
        for word in text.split(" "):
            words.append(word)

    print("长度均值 :", statistics.mean(length))
    print("长度中位数 :", statistics.median(length))
    print("长度方差 :", statistics.stdev(length))
    print("长度最小值：{}, 长度最大值：{}".format(min(length), max(length)))

    length = sorted(Counter(length).items(), key=lambda val: val[0], reverse=False)
    dict_sentence = sorted(Counter(query).items(), key=lambda val: val[1], reverse=True)
    dict_words = sorted(Counter(words).items(), key=lambda val: val[1], reverse=True)
    _, value_sentence = zip(*dict_sentence)
    _, value_word = zip(*dict_words)
    len_keys, value_len = zip(*length)

    plt.plot(range(1000), value_sentence[0:1000], 'r')
    plt.xlabel("words serial")
    plt.ylabel("number of words")
    plt.savefig("word_counter.png")
    plt.close()

    plt.plot(range(300), value_word[0:300], 'r')
    plt.xlabel("sentences serial")
    plt.ylabel("number of sentence")
    plt.savefig("sentence_counter.png")
    plt.close()

    plt.plot(len_keys[0:100], value_len[0:100], "r")
    plt.xlabel("句子长度")
    plt.ylabel("相同长度句子个数")
    plt.savefig("句子长度个数.png")
    plt.close()

    df_new = pd.DataFrame(data={"question1": query_a, "question2": query_b},
                          columns=[['question1', 'question2']])
    df_new.to_csv('test_new.csv', index=False)


def pre_split_train(out_feature):
    df = pd.read_csv("./data/csv/train.csv").dropna()
    data = df[['question1', 'question2', 'is_duplicate']].values
    # 获取句子的extra feature
    outer_feature = np.array(pickle.load(open(out_feature, "rb")))

    # random.shuffle(data)
    data_x = data[0:len(data), 0:2]
    data_y = data[0:len(data), 2:3]
    test_x, train_x = data_x[0:5000, :], data_x[5000:len(data_x), :]
    test_y, train_y = data_y[0:5000, :], data_y[5000:len(data_y), :]
    feature_test, feature_train = outer_feature[0:5000, :], outer_feature[5000:len(outer_feature), :]

    lcs_test, lcs_train = outer_feature[0:5000, 13:14], outer_feature[5000:len(outer_feature), 13:14]
    edit_distance_test, edit_distance_train = outer_feature[0:5000, 0:1], outer_feature[5000:len(outer_feature), 0:1]

    sentiment1_test, sentiment1_train = outer_feature[0:5000, 1:2], outer_feature[5000:len(outer_feature), 1:2]
    sentiment2_test, sentiment2_train = outer_feature[0:5000, 2:3], outer_feature[5000:len(outer_feature), 2:3]
    sentiment3_test, sentiment3_train = outer_feature[0:5000, 3:4], outer_feature[5000:len(outer_feature), 3:4]
    sentiment4_test, sentiment4_train = outer_feature[0:5000, 4:5], outer_feature[5000:len(outer_feature), 4:5]
    sentiment5_test, sentiment5_train = outer_feature[0:5000, 5:6], outer_feature[5000:len(outer_feature), 5:6]
    sentiment6_test, sentiment6_train = outer_feature[0:5000, 6:7], outer_feature[5000:len(outer_feature), 6:7]

    ratio1_test, ratio1_train = outer_feature[0:5000, 7:8], outer_feature[5000:len(outer_feature), 7:8]
    ratio2_test, ratio2_train = outer_feature[0:5000, 8:9], outer_feature[5000:len(outer_feature), 8:9]
    ratio3_test, ratio3_train = outer_feature[0:5000, 9:10], outer_feature[5000:len(outer_feature), 9:10]
    ratio4_test, ratio4_train = outer_feature[0:5000, 10:11], outer_feature[5000:len(outer_feature), 10:11]
    ratio5_test, ratio5_train = outer_feature[0:5000, 11:12], outer_feature[5000:len(outer_feature), 11:12]
    ratio6_test, ratio6_train = outer_feature[0:5000, 12:13], outer_feature[5000:len(outer_feature), 12:13]
    length_difference1_test, length_difference1_train = outer_feature[0:5000, 14:15], outer_feature[5000:len(outer_feature), 14:15]
    length_difference1_test, length_difference1_train = outer_feature[0:5000, 15:16], outer_feature[5000:len(outer_feature), 15:16]
    tf_idf_word_match_test, tf_idf_word_match_test = outer_feature[0:5000, 16:17], outer_feature[5000:len(outer_feature), 16:17]

    print("********************************************")
    print(feature_test.shape)
    print(feature_train.shape)
    print(data_x.shape)
    print("********************************************")

    test_df = pd.DataFrame(data={'question1': np.squeeze(test_x[:, 0:1], axis=1),
                                 'question2': np.squeeze(test_x[:, 1:2], axis=1),
                                 'lcs_test': np.squeeze(outer_feature[0:5000, 13:14], axis=1),
                                 'edit_distance_test': np.squeeze(outer_feature[0:5000, 0:1], axis=1),
                                 'sentiment1_test': np.squeeze(outer_feature[0:5000, 1:2], axis=1),
                                 'sentiment2_test': np.squeeze(outer_feature[0:5000, 2:3], axis=1),
                                 'sentiment3_test': np.squeeze(outer_feature[0:5000, 3:4], axis=1),
                                 'sentiment4_test': np.squeeze(outer_feature[0:5000, 4:5], axis=1),
                                 'sentiment5_test': np.squeeze(outer_feature[0:5000, 5:6], axis=1),
                                 'sentiment6_test': np.squeeze(outer_feature[0:5000, 6:7], axis=1),
                                 'ratio1_test': np.squeeze(outer_feature[0:5000, 7:8], axis=1),
                                 'ratio2_test': np.squeeze(outer_feature[0:5000, 8:9], axis=1),
                                 'ratio3_test': np.squeeze(outer_feature[0:5000, 9:10], axis=1),
                                 'ratio4_test': np.squeeze(outer_feature[0:5000, 10:11], axis=1),
                                 'ratio5_test': np.squeeze(outer_feature[0:5000, 11:12], axis=1),
                                 'ratio6_test': np.squeeze(outer_feature[0:5000, 12:13], axis=1),
                                 'length_difference1_test': np.squeeze(outer_feature[0:5000, 14:15], axis=1),
                                 'length_difference2_test': np.squeeze(outer_feature[0:5000, 15:16], axis=1),
                                 'tf_idf_word_match_test': np.squeeze(outer_feature[0:5000, 16:17], axis=1),
                                 'is_duplicate': np.squeeze(test_y, axis=1)},
                           columns=[columns])

    train_df = pd.DataFrame(data={'question1': np.squeeze(train_x[:, 0:1], axis=1),
                                  'question2': np.squeeze(train_x[:, 1:2], axis=1),
                                  'lcs_test': np.squeeze(outer_feature[5000:len(outer_feature), 13:14], axis=1),
                                  'edit_distance_test': np.squeeze(outer_feature[5000:len(outer_feature), 0:1], axis=1),
                                  'sentiment1_test': np.squeeze(outer_feature[5000:len(outer_feature), 1:2], axis=1),
                                  'sentiment2_test': np.squeeze(outer_feature[5000:len(outer_feature), 2:3], axis=1),
                                  'sentiment3_test': np.squeeze(outer_feature[5000:len(outer_feature), 3:4], axis=1),
                                  'sentiment4_test': np.squeeze(outer_feature[5000:len(outer_feature), 4:5], axis=1),
                                  'sentiment5_test': np.squeeze(outer_feature[5000:len(outer_feature), 5:6], axis=1),
                                  'sentiment6_test': np.squeeze(outer_feature[5000:len(outer_feature), 6:7], axis=1),
                                  'ratio1_test': np.squeeze(outer_feature[5000:len(outer_feature), 7:8], axis=1),
                                  'ratio2_test': np.squeeze(outer_feature[5000:len(outer_feature), 8:9], axis=1),
                                  'ratio3_test': np.squeeze(outer_feature[5000:len(outer_feature), 9:10], axis=1),
                                  'ratio4_test': np.squeeze(outer_feature[5000:len(outer_feature), 10:11], axis=1),
                                  'ratio5_test': np.squeeze(outer_feature[5000:len(outer_feature), 11:12], axis=1),
                                  'ratio6_test': np.squeeze(outer_feature[5000:len(outer_feature), 12:13], axis=1),
                                  'length_difference1_test': np.squeeze(outer_feature[5000:len(outer_feature), 14:15], axis=1),
                                  'length_difference2_test': np.squeeze(outer_feature[5000:len(outer_feature), 15:16], axis=1),
                                  'tf_idf_word_match_test': np.squeeze(outer_feature[5000:len(outer_feature), 16:17], axis=1),
                                  'is_duplicate': np.squeeze(train_y, axis=1)},
                            columns=[columns])

    print("写入csv数据。。。")
    test_df.to_csv("./data/csv/train_test.csv", columns=columns)
    train_df.to_csv("./data/csv/train_train.csv", columns=columns)
    print("写入csv数据成功！！！")


def remove_stop_words(sentence, stop_words_set):
    ans = []
    for word in sentence.split():
        if word.lower() not in stop_words_set:
            ans.append(word)

    return " ".join(ans)


class data(object):
    def __init__(self, train_file_path, test_file_path, stop_words_file):
        self.path = os.path.dirname(__file__)
        self.stop_words = set(open(stop_words_file, "r").read().split())

        # 获取训练数据，数据来源于 train_file_path
        self.df = pd.read_csv(train_file_path).dropna()
        self.data = self.df[['question1', 'question2']].values
        self.label = self.df[['is_duplicate']].values
        self.train_feature = self.df[columns_extra_feature].values

        # 获取测试数据，数据来源于 train_file_path
        self.test_df = pd.read_csv(test_file_path).dropna()
        self.test_data = self.test_df[['question1', 'question2']].values
        self.test_label = self.test_df[['is_duplicate']].values
        self.test_feature = self.test_df[columns_extra_feature].values

        print(datetime.datetime.now().isoformat())
        print("当前文件路径 :", self.path)
        print("self.data.shape :", self.data.shape)
        print("self.label.shape :", self.label.shape)
        print("self.train_feature.shape :", self.train_feature.shape)
        print("self.test_data.shape :", self.test_data.shape)
        print("self.test_label.shape :", self.test_label.shape)
        print("self.test_feature.shape :", self.test_feature.shape)


    @staticmethod
    def text_to_wordlist(text):
        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(r"what's", "", text)
        text = re.sub(r"What's", "", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"60k", " 60000 ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e-mail", "email", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"quikly", "quickly", text)
        text = re.sub(r" usa ", " America ", text)
        text = re.sub(r" USA ", " America ", text)
        text = re.sub(r" u s ", " America ", text)
        text = re.sub(r" uk ", " England ", text)
        text = re.sub(r" UK ", " England ", text)
        text = re.sub(r"india", "India", text)
        text = re.sub(r"switzerland", "Switzerland", text)
        text = re.sub(r"china", "China", text)
        text = re.sub(r"chinese", "Chinese", text)
        text = re.sub(r"imrovement", "improvement", text)
        text = re.sub(r"intially", "initially", text)
        text = re.sub(r"quora", "Quora", text)
        text = re.sub(r" dms ", "direct messages ", text)
        text = re.sub(r"demonitization", "demonetization", text)
        text = re.sub(r"actived", "active", text)
        text = re.sub(r"kms", " kilometers ", text)
        text = re.sub(r"KMs", " kilometers ", text)
        text = re.sub(r" cs ", " computer science ", text)
        text = re.sub(r" upvotes ", " up votes ", text)
        text = re.sub(r" iPhone ", " phone ", text)
        text = re.sub(r"\0rs ", " rs ", text)
        text = re.sub(r"calender", "calendar", text)
        text = re.sub(r"ios", "operating system", text)
        text = re.sub(r"gps", "GPS", text)
        text = re.sub(r"gst", "GST", text)
        text = re.sub(r"programing", "programming", text)
        text = re.sub(r"bestfriend", "best friend", text)
        text = re.sub(r"dna", "DNA", text)
        text = re.sub(r"III", "3", text)
        text = re.sub(r"the US", "America", text)
        text = re.sub(r"Astrology", "astrology", text)
        text = re.sub(r"Method", "method", text)
        text = re.sub(r"Find", "find", text)
        text = re.sub(r"banglore", "Banglore", text)
        text = re.sub(r" J K ", " JK ", text)
        return text

    def get_one_hot(self):
        if not os.path.exists(os.path.join(self.path, "data/pkl/test.pkl")):
            x_text = np.append(self.data, self.test_data).reshape(2 * len(self.data) + 2 * len(self.test_data)) # 所有的文本数据

            # reshape 数据
            self.data = self.data.reshape(2 * len(self.data))
            self.test_data = self.test_data.reshape(2 * len(self.test_data))

            # 清洗数据
            self.data = [self.text_to_wordlist(line) for line in self.data]
            self.test_data = [self.text_to_wordlist(line) for line in self.test_data]

            # 转化成数据，将词汇进行编号
            vocab_processor = learn.preprocessing.VocabularyProcessor(50, min_frequency=5)
            vocab_processor = vocab_processor.fit(x_text)
            print("vocab_processor 训练结束")

            # 训练数据和测试数据进行编号
            self.vec_train = list(vocab_processor.transform(self.data))
            self.vec_test = list(vocab_processor.transform(self.test_data))

            # 编号
            self.vec_train = [(self.vec_train[index], self.vec_train[index + 1]) for index in range(0, len(self.vec_train), 2)]
            self.vec_test = [(self.vec_test[index], self.vec_test[index + 1]) for index in range(0, len(self.vec_test), 2)]
            print("vocab_processor 转化结束")

            context_ids = [list(range(len(vocab_processor.vocabulary_)))]
            # print("number of words :", len(vocab_processor.vocabulary_))
            # print(vocab_processor.reverse(context_ids))
            # for article in vocab_processor.reverse(context_ids):
            #     for word in article.split():
            #         print(word)

            pickle.dump((vocab_processor), open(os.path.join(self.path, "data/vocab.model"), "wb"))
            pickle.dump((self.vec_train, self.label), open(os.path.join(self.path, "data/pkl/train.pkl"), "wb"))
            pickle.dump((self.vec_test, self.test_label), open(os.path.join(self.path, "data/pkl/test.pkl"), "wb"))
            pickle.dump(list(vocab_processor.reverse(context_ids)), open(os.path.join(self.path, "data/pkl/bag.pkl"), "wb"))
            pickle.dump((self.test_feature, self.train_feature), open(os.path.join(self.path, "data/pkl/extra_feature.pkl"), "wb"))
            print("number of words :", len(vocab_processor.vocabulary_))
            print("dump 结束")
        else:
            self.vec_train, self.label = pickle.load(open(os.path.join(self.path, "data/pkl/train.pkl"), "rb"))
            self.vec_test, self.test_label = pickle.load(open(os.path.join(self.path, "data/pkl/test.pkl"), "rb"))
            self.test_feature, self.train_feature = pickle.load(open(os.path.join(self.path, "data/pkl/extra_feature.pkl"), "rb"))
        return self

    @staticmethod
    def get_batch(epoches, batch_size, data, out_feature, label):
        data = list(zip(data, out_feature, label))
        for epoch in range(epoches):
            random.shuffle(data)
            for batch in range(0, len(data), batch_size):
                if batch + batch_size >= len(data):
                    yield data[batch: len(data)]
                else:
                    yield data[batch: (batch + batch_size)]


if __name__ == '__main__':
    # pre_split_train()
    data_file = "./data/csv/train.csv"
    train_file = "./data/csv/train_train.csv"
    test_file = "./data/csv/train_test.csv"
    stop_words_file = "./data/stop_words_eng.txt"
    pre_split_train("./data/feature.pkl")
