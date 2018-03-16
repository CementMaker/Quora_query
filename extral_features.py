import re
import os
import pickle
import random
import statistics

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.externals import joblib

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def remove_stop_words(sentence, stop_words_set):
    ans = []
    for word in sentence.split():
        if word.lower() not in stop_words_set:
            ans.append(word)

    return " ".join(ans)


class sentiment(object):
    def __init__(self):
        self.df = pd.read_csv("./data/csv/Tweets.csv")
        self.data = np.squeeze(self.df[['text']].values, axis=1)
        self.label = self.df[['airline_sentiment', 'airline_sentiment_confidence']].values
        self.value = [0] * len(self.label)

        for index in range(len(self.label)):
            if self.label[index][0] == 'neutral':
                # self.value[index] = 0.5
                self.value[index] = 0
            elif self.label[index][0] == 'negative':
                # self.value[index] = (-1.0 * self.label[index][1] + 1) / 2
                self.value[index] = -1
            else:
                # self.value[index] = (self.label[index][1] + 1) / 2
                self.value[index] = 1

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data, np.array(self.value), test_size=0.05)
        self.vectorizer = CountVectorizer(
            max_df=0.5,
            max_features=3000,
            min_df=3,
            lowercase=False,
            decode_error='ignore').fit(self.data)

        self.train_x = self.vectorizer.transform(self.train_x).toarray()
        self.test_x = self.vectorizer.transform(self.test_x).toarray()

    def xgbRegressionModel(self):
        regr = xgb.XGBClassifier(
            n_estimators=200,
            colsample_bytree=0.4,
            gamma=0.05,
            learning_rate=0.16,
            max_depth=6,
            reg_alpha=1.2,
            subsample=1)

        regr.fit(self.train_x, self.train_y)
        y_pred_xgb = regr.predict(self.test_x)
        print(classification_report(y_true= np.expand_dims(self.test_y, axis=-1),
                                    y_pred=y_pred_xgb))
        joblib.dump(regr, './data/xgb_sentiment.model')

    def logisticRegression(self):
        print(self.train_x.shape)
        print(self.train_y.shape)
        self.lr = LogisticRegression(n_jobs=4).fit(self.train_x, self.train_y)
        y_pred_lr = self.lr.predict(self.test_x)
        print(classification_report(y_true=np.expand_dims(self.test_y, axis=-1),
                                    y_pred=y_pred_lr))
        joblib.dump(self.lr, "./data/lr_sentiment.model")


class ManualFeatureExtraction(object):
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file).dropna()[['question1', 'question2']]
        self.corpus = np.reshape(a=self.df.values,
                                 newshape=len(self.df.values) * 2)

        self.vectorizer = TfidfVectorizer(
            max_df=0.5,
            max_features=3000,
            min_df=1,
            use_idf=True,
            lowercase=False,
            decode_error='ignore',
        ).fit(self.corpus)


        print(self.df.values.shape)
        self.train_document = self.df.values
        self.lr = joblib.load("./data/lr_sentiment.model")

    def tf_idf_word_match(self, sentencea, sentenceb):
        sentencea = sentencea.split()
        sentenceb = sentenceb.split()
        match = " ".join(list(set(sentencea) & set(sentenceb)))
        combine = " ".join(list(set(sentencea) | set(sentenceb)))

        if(len(match) == 0): return 0.
        tf_idf_a = self.vectorizer.transform([match]).toarray()[0]
        tf_idf_b = self.vectorizer.transform([combine]).toarray()[0]
        return sum(tf_idf_a) / sum(tf_idf_b)

    @staticmethod
    def length_difference(sentencea, sentenceb):
        return len(sentencea) - len(sentenceb), len(sentencea.split()) - len(sentenceb.split())

    @staticmethod
    def LongCommonSequence(sentencea, sentenceb):
        sentencea = sentencea.split()
        sentenceb = sentenceb.split()
        lena, lenb = len(sentencea), len(sentenceb)
        dp = np.array([[0] * (lenb + 1) for _ in range(lena + 1)])

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
        number = 0
        self.outer_feature = []
        for line in self.train_document:
            # 情感分析
            sentencea, sentenceb = line
            tmp = self.vectorizer.transform(line)

            sentiment = list(self.lr.predict_proba(tmp).flatten())
            ratio = list(self.fuzzy_ratio(sentencea, sentenceb))
            edit_distance = [self.edit_distance_word(sentencea, sentenceb)]
            lcs = [self.LongCommonSequence(sentencea, sentenceb)]
            length_difference = list(self.length_difference(sentencea, sentenceb))
            tf_idf_word_match = [self.tf_idf_word_match(sentencea, sentenceb)]

            # self.lcs.append(lcs)
            # self.ratio.append(ratio)
            # self.sentiment.append(sentiment)
            # self.edit_distance.append(edit_distance)
            # self.length_diff.append(length_difference)

            tmp = edit_distance + sentiment + ratio + lcs + length_difference + tf_idf_word_match
            self.outer_feature.append(tmp)

            number += 1
            if number % 5000 == 0:
                datetimestr = datetime.datetime.now().isoformat()
                print(datetimestr, number, "lines processed", np.array(self.outer_feature).shape)

        print(datetime.datetime.now().isoformat())


if __name__ == '__main__':
    # pre_split_train()
    data_file = "./data/csv/train.csv"
    train_file = "./data/csv/train_train.csv"
    test_file = "./data/csv/train_test.csv"
    stop_words_file = "./data/stop_words_eng.txt"

    # sentiment().xgbRegressionModel()
    # sentiment().logisticRegression()

    # Data = data(train_file, test_file, stop_words_file)
    # Data = data(train_file, test_file, stop_words_file).get_one_hot()
    print(datetime.datetime.now().isoformat())
    ManualFeatureExtraction(data_file).main()
    # feature = ManualFeatureExtraction(data_file)
    # feature.main()

