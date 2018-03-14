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

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

path = os.getcwd()


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


def pre_split_train():
    df = pd.read_csv("./data/csv/train.csv").dropna()
    data = df[['question1', 'question2', 'is_duplicate']].values

    # random.shuffle(data)
    data_x = data[0:len(data), 0:2]
    data_y = data[0:len(data), 2:3]
    test_x, train_x = data_x[0:5000, :], data_x[5000:len(data_x), :]
    test_y, train_y = data_y[0:5000, :], data_y[5000:len(data_y), :]

    test_df = pd.DataFrame(data={'question1': np.squeeze(test_x[:, 0:1], axis=1),
                                 'question2': np.squeeze(test_x[:, 1:2], axis=1),
                                 'is_duplicate': np.squeeze(test_y, axis=1)},
                           columns=[['question1', 'question2', 'is_duplicate']])

    train_df = pd.DataFrame(data={'question1': np.squeeze(train_x[:, 0:1], axis=1),
                                  'question2': np.squeeze(train_x[:, 1:2], axis=1),
                                  'is_duplicate': np.squeeze(train_y, axis=1)},
                            columns=[['question1', 'question2', 'is_duplicate']])

    print("写入csv数据。。。")
    test_df.to_csv("./data/csv/train_test.csv", columns=['question1', 'question2', 'is_duplicate'])
    train_df.to_csv("./data/csv/train_train.csv", columns=['question1', 'question2', 'is_duplicate'])
    print("写入csv数据成功！！！")


def remove_stop_words(sentence, stop_words_set):
    ans = []
    for word in sentence.split():
        if word.lower() not in stop_words_set:
            ans.append(word)

    return " ".join(ans)


class ManualFeatureExtraction(object):
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file).dropna()[['question1', 'question2']]
        self.corpus = np.reshape(a=self.df.values,
                                 newshape=len(self.df.values) * 2)
        print(self.corpus.shape)
        self.vectorizer = TfidfVectorizer(
            max_df=0.5,
            max_features=5000,
            min_df=1,
            use_idf=True,
            lowercase=False,
            decode_error='ignore',
        ).fit(self.corpus)
        self.train_document = self.df.values


    def tf_idf_word_match(self, sentencea, sentenceb):
        sentencea = sentencea.split()
        sentenceb = sentenceb.split()
        match = set(sentencea) & set(sentenceb)
        combine = set(sentencea) | set(sentenceb)

        if(len(match) == 0): return 0.
        tf_idf_a = self.vectorizer.transform(match).toarray()[0]
        tf_idf_b = self.vectorizer.transform(combine).toarray()[0]
        return sum(tf_idf_a) / sum(tf_idf_b)

    def sentiment(self, sentence):
        pass

    @staticmethod
    def length_difference(sentencea, sentenceb):
        return len(sentencea) - len(sentenceb), len(sentencea.split()) - len(sentenceb.split())

    @staticmethod
    def LongCommonSequence(sentencea, sentenceb):
        sentencea = sentencea.split()
        sentenceb = sentenceb.split()
        lena, lenb = len(sentencea), len(sentenceb)
        dp = np.array([[0] * (lena + 1) for _ in range(lenb + 1)])

        for i in range(lena):
            for j in range(lenb):
                if sentencea[i] == sentenceb[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])

        return dp[lena][lenb]

    @staticmethod
    def edit_distance_word(sentencea, sentenceb):
        sentencea = sentencea.split()
        sentenceb = sentenceb.split()
        print(sentencea)
        print(sentenceb)
        lena, lenb = len(sentencea), len(sentenceb)
        dp = [[0] * (lenb + 1) for _ in range(lena + 1)]

        # 长度为i句子变换成长度为0的句子的编辑距离
        # 下标为i，子列表的长度为i + 1,所以有 dp[i + 1][0] = i + 1
        for i in range(lena + 1): dp[i][0] = i
        for j in range(lenb + 1): dp[0][j] = j

        for i in range(lena):
            for j in range(lenb):
                if sentencea[i] == sentenceb[j]:
                    dp[i + 1][j + 1] = dp[i][j]
                else:
                    dp[i + 1][j + 1] = min(dp[i + 1][j], dp[i][j + 1], dp[i][j]) + 1
            #     print(sentencea[0 : i + 1], sentenceb[0 : j + 1], dp[i + 1][j + 1])
            # print()

        return dp[lena][lenb]

    @staticmethod
    def fuzzy_ratio(sentencea, sentenceb):
        ratio = fuzz.ratio(sentencea, sentenceb)
        partial_ratio = fuzz.partial_ratio(sentencea, sentenceb)
        token_sort_ratio = fuzz.token_sort_ratio(sentencea, sentenceb)
        token_set_ratio = fuzz.token_set_ratio(sentencea, sentenceb)
        partial_token_set_ratio = fuzz.partial_token_set_ratio(sentencea, sentenceb)
        partial_token_sort_ratio = fuzz.partial_token_sort_ratio(sentencea, sentenceb)
        return ratio, partial_ratio, token_set_ratio, token_sort_ratio, partial_token_set_ratio, partial_token_sort_ratio

    def main(self):
        for a, b in self.train_document:
            print(a, b)
            print(self.tf_idf_word_match(a, b))


class data(object):
    def __init__(self, train_file_path, test_file_path, stop_words_file):

        # 获取训练数据，数据来源于 train_file_path
        self.df = pd.read_csv(train_file_path).dropna()
        self.path = os.path.dirname(__file__)
        self.columns = ['question1', 'question2', 'is_duplicate']
        self.stop_words = set(open(stop_words_file, "r").read().split())

        number = 0
        print(datetime.datetime.now().isoformat())
        self.data = self.df[['question1', 'question2']].values
        # for index in range(len(self.data)):
        #     sentencea, sentenceb = self.data[index]
        #     self.data[index][0] = remove_stop_words(sentencea, self.stop_words)
        #     self.data[index][1] = remove_stop_words(sentenceb, self.stop_words)
        #
        #     number += 1
        #     if number % 10000 == 0:
        #         print("number :", number, datetime.datetime.now().isoformat())
        #         print(self.data[index][0], self.data[index][1])

        self.label = self.df[['is_duplicate']].values

        print("当前文件路径 :", self.path)
        print("self.data.shape :", self.data.shape)
        print("self.label.shape :", self.label.shape)

        self.test_df = pd.read_csv(test_file_path).dropna()
        self.test_data = self.test_df[['question1', 'question2']].values
        self.test_label = self.test_df[['is_duplicate']].values

        print("self.test_data.shape :", self.test_data.shape)
        print("self.test_label.shape :", self.test_label.shape)


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
            x_text = np.append(self.data, self.test_data).reshape(2 * len(self.data) + 2 * len(self.test_data))
            self.data = self.data.reshape(2 * len(self.data))
            self.test_data = self.test_data.reshape(2 * len(self.test_data))
            self.data = [self.text_to_wordlist(line) for line in self.data]
            self.test_data = [self.text_to_wordlist(line) for line in self.test_data]

            vocab_processor = learn.preprocessing.VocabularyProcessor(50, min_frequency=5)
            vocab_processor = vocab_processor.fit(x_text)
            print("vocab_processor 训练结束")

            self.vec_train = list(vocab_processor.transform(self.data))
            self.vec_test = list(vocab_processor.transform(self.test_data))
            self.vec_train = [(self.vec_train[index], self.vec_train[index + 1]) for index in range(0, len(self.vec_train), 2)]
            self.vec_test = [(self.vec_test[index], self.vec_test[index + 1]) for index in range(0, len(self.vec_test), 2)]
            print("vocab_processor 转化结束")

            context_ids = [list(range(len(vocab_processor.vocabulary_)))]
            print("number of words :", len(vocab_processor.vocabulary_))
            print(vocab_processor.reverse(context_ids))
            # for article in vocab_processor.reverse(context_ids):
            #     for word in article.split():
            #         print(word)

            pickle.dump((vocab_processor), open(os.path.join(self.path, "data/vocab.model"), "wb"))
            pickle.dump((self.vec_train, self.label), open(os.path.join(self.path, "data/pkl/train.pkl"), "wb"))
            pickle.dump((self.vec_test, self.test_label), open(os.path.join(self.path, "data/pkl/test.pkl"), "wb"))
            pickle.dump(list(vocab_processor.reverse(context_ids)), open(os.path.join(self.path, "data/pkl/bag.pkl"), "wb"))
            print("number of words :", len(vocab_processor.vocabulary_))
            print("dump 结束")
        else:
            self.vec_train, self.label = pickle.load(open(os.path.join(self.path, "data/pkl/train.pkl"), "rb"))
            self.vec_test, self.test_label = pickle.load(open(os.path.join(self.path, "data/pkl/test.pkl"), "rb"))
        return self

    @staticmethod
    def get_batch(epoches, batch_size, data, label):
        data = list(zip(data, label))
        for epoch in range(epoches):
            random.shuffle(data)
            for batch in range(0, len(data), batch_size):
                if batch + batch_size >= len(data):
                    yield data[batch: len(data)]
                else:
                    yield data[batch: (batch + batch_size)]


    @staticmethod
    def magic_feature():
        pass



class data_create(object):
    def __init__(self, train_file):
        self.columns = ['question1', 'question2', 'is_duplicate']

        # 训练数据
        self.df = pd.read_csv(train_file).dropna()
        self.df = pd.concat([pd.DataFrame(data={"question1": self.df["question2"],
                                                "question2": self.df["question1"],
                                                "is_duplicate": self.df["is_duplicate"]},
                                          columns=["question1", "question2", 'is_duplicate']), self.df],
                            axis=0, ignore_index=True)

        self.data = self.df[['question1', 'question2']].values
        self.label = self.df[['is_duplicate']].values

        self.train_data, self.test_data, self.train_label, self.test_label =\
            train_test_split(self.data, self.label, test_size=0.01)
        self.train_data, self.train_label = self.resample(self.train_data, self.train_label)
        self.test_data, self.test_label = self.resample(self.test_data, self.test_label)
        self.path = os.path.dirname(__file__)

        print("当前文件路径 :", self.path)
        print("self.data.shape :", self.data.shape)
        print("self.label.shape :", self.label.shape)
        print("self.test_data.shape :", self.test_data.shape)
        print("self.test_label.shape :", self.test_label.shape)

    @staticmethod
    def resample(data, label):
        data_double, label_double = [], []
        for (d, l) in zip(data, label):
            if l == 0 and random.random() <= 0.5667:
                data_double.append(d)
                label_double.append(l)

        data = np.append(data, data_double, axis=0)
        label = np.append(label, label_double)
        return data, label

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
        if not os.path.exists(os.path.join(self.path, "data/pkl/train.pkl")):
            self.data = self.data.reshape(2 * len(self.data))
            self.test_data = self.test_data.reshape(2 * len(self.test_data))
            self.train_data = self.train_data.reshape(2 * len(self.train_data))
            self.data = [self.text_to_wordlist(line) for line in self.data]
            self.test_data = [self.text_to_wordlist(line) for line in self.test_data]
            self.train_data = [self.text_to_wordlist(line) for line in self.train_data]

            vocab_processor = learn.preprocessing.VocabularyProcessor(50, min_frequency=5)
            vocab_processor = vocab_processor.fit(self.data)
            print("vocab_processor 训练结束")

            self.vec_train = list(vocab_processor.transform(self.train_data))
            self.vec_test = list(vocab_processor.transform(self.test_data))
            self.vec_train = [(self.vec_train[index], self.vec_train[index + 1]) for index in range(0, len(self.vec_train), 2)]
            self.vec_test = [(self.vec_test[index], self.vec_test[index + 1]) for index in range(0, len(self.vec_test), 2)]
            print("vocab_processor 转化结束")

            context_ids = [list(range(len(vocab_processor.vocabulary_)))]
            print("number of words :", len(vocab_processor.vocabulary_))
            print(vocab_processor.reverse(context_ids))
            # for article in vocab_processor.reverse(context_ids):
            #     for word in article.split():
            #         print(word)

            pickle.dump((vocab_processor), open(os.path.join(self.path, "data/vocab.model"), "wb"))
            pickle.dump((self.vec_train, self.label), open(os.path.join(self.path, "data/pkl/train.pkl"), "wb"))
            pickle.dump((self.vec_test, self.test_label), open(os.path.join(self.path, "data/pkl/test.pkl"), "wb"))
            pickle.dump(list(vocab_processor.reverse(context_ids)), open(os.path.join(self.path, "data/pkl/bag.pkl"), "wb"))
            print("number of words :", len(vocab_processor.vocabulary_))
            print("dump 结束")
        else:
            self.vec_train, self.label = pickle.load(open(os.path.join(self.path, "data/pkl/train.pkl"), "rb"))
            self.vec_test, self.test_label = pickle.load(open(os.path.join(self.path, "data/pkl/test.pkl"), "rb"))
        return self

    @staticmethod
    def get_batch(epoches, batch_size, data, label):
        data = list(zip(data, label))
        for epoch in range(epoches):
            random.shuffle(data)
            for batch in range(0, len(data), batch_size):
                if batch + batch_size >= len(data):
                    yield data[batch: len(data)]
                else:
                    yield data[batch: (batch + batch_size)]


    @staticmethod
    def magic_feature():
        pass


if __name__ == '__main__':
    # pre_split_train()
    data_file = "./data/csv/train.csv"
    train_file = "./data/csv/train_train.csv"
    test_file = "./data/csv/train_test.csv"
    stop_words_file = "./data/stop_words_eng.txt"

    # Data = data(train_file, test_file, stop_words_file)
    Data = data(train_file, test_file, stop_words_file).get_one_hot()

    # ManualFeatureExtraction(data_file, train_file, test_file)
    # feature = ManualFeatureExtraction(data_file)
    # feature.main()

