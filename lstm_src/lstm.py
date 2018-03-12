#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import rnn


class Model(object):
    def BiRNN(self, x, nscope, vscope, embedding_size, rnn_size, sequence_length, num_layer):
        n_input, n_steps, n_hidden, n_layers = embedding_size, sequence_length, rnn_size, num_layer
        with tf.name_scope("fw" + nscope), tf.variable_scope("fw" + vscope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
                # stm_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=1)
                stacked_rnn_fw.append(fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + nscope), tf.variable_scope("bw" + vscope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
                # stm_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=1)
                stacked_rnn_bw.append(bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.name_scope("bw" + nscope), tf.variable_scope("bw" + vscope):
            outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell_m,
                                                         cell_bw=lstm_bw_cell_m,
                                                         inputs=x,
                                                         dtype=tf.float32,
                                                         scope=nscope)
        return outputs[-1]

    def lstm(self, input, rnn_size, num_layers, scope):
        stack_lstm = []
        for _ in range(num_layers):
            cell = rnn.BasicLSTMCell(rnn_size)
            stm_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
            stack_lstm.append(stm_cell)
        lstm_cell_m = rnn.MultiRNNCell(stack_lstm, state_is_tuple=True)

        # stack_lstm = [rnn.BasicLSTMCell(rnn_size)] * num_layers
        # lstm_cell_m = rnn.MultiRNNCell(stack_lstm, state_is_tuple=True)

        outputs, _ = tf.nn.dynamic_rnn(lstm_cell_m, input, dtype=tf.float32, scope=scope)
        return outputs[-1]

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def __init__(self, num_layers, seq_length, embedding_size, vocab_size, rnn_size, batch_size):
        # 输入数据以及数据标签
        self.label = tf.placeholder(tf.float32, [None, ], name="label")
        self.input_sentence_a = tf.placeholder(tf.int32, [None, seq_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, seq_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_loss = tf.constant(0.0)

        with tf.name_scope('embeddingLayer'):
            # W : 词表（embedding 向量），后面用来训练.
            w = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            embedded_a = tf.nn.embedding_lookup(w, self.input_sentence_a)
            embedded_b = tf.nn.embedding_lookup(w, self.input_sentence_b)

            inputs_a = tf.unstack(embedded_a, axis=1)
            inputs_b = tf.unstack(embedded_b, axis=1)

        # outputs是最后一层每个节点的输出
        # last_state是每层最后一个节点的输出。
        with tf.name_scope("output"):
            # lstm 共享权重
            # self.out1 = self.lstm(inputs_a, rnn_size, num_layers, scope="a")
            # self.out2 = self.lstm(inputs_b, rnn_size, num_layers, scope="b")

            # lstm 没有共享权重
            # self.out1 = self.lstm(inputs_a, rnn_size, num_layers, scope="a")
            # self.out2 = self.lstm(inputs_b, rnn_size, num_layers, scope="b")

            # 双向lstm，没有共享权重
            # self.out1 = self.BiRNN(inputs_a, "a", "a", embedding_size, rnn_size, seq_length, num_layers)
            # self.out2 = self.BiRNN(inputs_b, "b", "b, embedding_size, rnn_size, seq_length, num_layers)

            # 双向lstm，共享权重
            self.out1 = self.BiRNN(inputs_a, "a", "a", embedding_size, rnn_size, seq_length, num_layers)
            self.out2 = self.BiRNN(inputs_b, "b", "b", embedding_size, rnn_size, seq_length, num_layers)

        with tf.name_scope("scope"):
            self.diff = self.out1 - self.out2
            self.mul = tf.multiply(self.out1, self.out1)

            self.feature = tf.concat([self.diff, self.mul, self.out1, self.out1], axis=1)
            self.weight = tf.Variable(tf.truncated_normal(shape=[rnn_size * 8, 1],
                                                          stddev=0.1,
                                                          mean=0.0))
            self.bias = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1, mean=0.0))
            self.result = tf.nn.xw_plus_b(self.feature, self.weight, self.bias)
            self.logits = tf.nn.sigmoid(self.result)
            self.ans = tf.squeeze(self.logits, axis=1)

        with tf.name_scope("loss"):
            # log_loss
            # tf.nn.softmax_cross_entropy_with_logits:
            #   先计算logits的softmax函数值，如果Logits就是一个概率分布，会带来loss并不是实际loss的问题
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weight) + tf.nn.l2_loss(self.bias)
            self.log_loss = tf.losses.log_loss(labels=self.label, predictions=self.ans)
            self.loss = self.log_loss# + 0.01 * self.l2_loss

        with tf.name_scope("accuracy"):
            self.predict = tf.rint(self.ans)
            self.tmp_sim = tf.equal(tf.cast(self.predict, tf.int64), tf.cast(self.label, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(self.tmp_sim, tf.float32))

    @staticmethod
    def logistic_regression(x, w, b):
        return 1 / (1 + tf.exp(-1 * tf.nn.xw_plus_b(x, w, b)))

    @staticmethod
    def contrastive_loss(y, d, batch_size):
        '''
        :param y: 训练样本label
        :param d: 神经网络训练出来的相似度
        :param batch_size: mini_batch的大小
        :return: 对比损失
        '''
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    @staticmethod
    def cosine_half_sita_square(feature_a, feature_b):
        '''
        :param feature_a: 特征a, shape = [?, length]
        :param feature_b: 特征b, shape = [?, length]
        :return: 两个向量的余弦距离（相似度）
        '''
        similarity = tf.reduce_sum(feature_a * feature_b, axis=1)
        similarity = tf.div(
            x=similarity,
            y=tf.multiply(tf.sqrt(tf.reduce_sum(feature_a * feature_a, axis=1)),
                          tf.sqrt(tf.reduce_sum(feature_b * feature_b, axis=1))))
        return (similarity + 1) / 2

    @staticmethod
    def euclidean_distance_normalization(feature_a, feature_b):
        '''
        :param feature_a: 特征a, shape = [?, length]
        :param feature_b: 特征b, shape = [?, length]
        :return: 两个向量归一化后的欧式距离（或者是 1 - 相似度）
        '''
        euclidean_distance = tf.square(tf.reduce_sum(tf.subtract(feature_a, feature_b), axis=1))
        euclidean_distance_norm = tf.div(euclidean_distance,
                                         tf.add(tf.sqrt(tf.reduce_sum(tf.square(feature_a), 1, keep_dims=True)),
                                                tf.sqrt(tf.reduce_sum(tf.square(feature_b), 1, keep_dims=True))))
        euclidean_distance_norm = tf.reshape(euclidean_distance_norm, [-1])
        return euclidean_distance_norm

    @staticmethod
    def euclidean_distance(feature_a, feature_b):
        '''
        :param feature_a: 特征a, shape = [?, length]
        :param feature_b: 特征b, shape = [?, length]
        :return: 两个向量的欧式距离（或者是 1 - 相似度）
        '''
        return tf.square(tf.reduce_sum(tf.subtract(feature_a, feature_b), axis=1))

    @staticmethod
    def accuracy(y_true, y_pred):
        '''
        :param y_true: 样本的实际标签, 必须为整数
        :param y_pred: 样本的预测标签, 整数或者浮点数都可以
        :return: 训练或者测试的准确率
        '''
        y_pred = tf.rint(y_pred)
        correct_predictions = tf.equal(y_true, y_pred)
        accuracy = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.float32))
        return accuracy


# lstm = Model(num_layers=3,
#              seq_length=60,
#              embedding_size=100,
#              vocab_size=35325,
#              rnn_size=50,
#              batch_size=500)



