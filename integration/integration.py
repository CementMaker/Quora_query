#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from tensorflow.contrib import rnn
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Model(object):
    def BiRNN(self, x, nscope, vscope, embedding_size, rnn_size, sequence_length, num_layer):
        '''
        :param x: 双向lstm的输入
        :param nscope: namespace scope
        :param vscope: variable scope
        :param embedding_size: embedding size
        :param rnn_size: rnn size
        :param sequence_length: 序列长度
        :param num_layer: 双向lstm长度
        :return: 返回最后一个输出
        '''
        n_input, n_steps, n_hidden, n_layers = embedding_size, sequence_length, rnn_size, num_layer
        with tf.name_scope("fw" + nscope), tf.variable_scope("fw" + vscope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
                stm_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=0.5)
                stacked_rnn_fw.append(fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + nscope), tf.variable_scope("bw" + vscope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
                stm_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=0.5)
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

        outputs, _ = tf.nn.dynamic_rnn(lstm_cell_m, input, dtype=tf.float32, scope=scope)
        return outputs[-1]

    def contrastive_loss(self, y, d, batch_size):
        '''
        :param y: 实际label
        :param d: 输出的distance
        :param batch_size: batch 的size大小
        :return: 对比损失
        '''
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters, num_layers, rnn_size, batch_size):
        self.label = tf.placeholder(tf.float32, [None, ], name="label")
        self.outer_feature = tf.placeholder(tf.float32, [None, 17], name="outer_feature")
        self.input_sentence_a = tf.placeholder(tf.int32, [None, sequence_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_loss = 0

        with tf.name_scope("embedding_layer"):
            self.W1 = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size],
                                                     dtype=tf.float32,
                                                     stddev=0.1,
                                                     mean=0.0), name="W")

            self.W2 = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size],
                                                     dtype=tf.float32,
                                                     stddev=0.1,
                                                     mean=0.0), name="W")

            self.embedded_a_cnn = tf.nn.embedding_lookup(params=self.W1, ids=self.input_sentence_a)
            self.embedded_b_cnn = tf.nn.embedding_lookup(params=self.W1, ids=self.input_sentence_b)

            self.embedded_a_rnn = tf.nn.embedding_lookup(params=self.W1, ids=self.input_sentence_a)
            self.embedded_b_rnn = tf.nn.embedding_lookup(params=self.W1, ids=self.input_sentence_b)

            # 循环神经网络的输入
            self.inputs_a = tf.unstack(self.embedded_a_rnn, axis=1)
            self.inputs_b = tf.unstack(self.embedded_b_rnn, axis=1)

            # CNN输入
            self.embedded_a_expand = tf.expand_dims(input=self.embedded_a_cnn, axis=-1)
            self.embedded_b_expand = tf.expand_dims(input=self.embedded_b_cnn, axis=-1)

        pooled_outputs_a = []
        pooled_outputs_b = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % i):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                w = tf.Variable(tf.truncated_normal(shape=filter_shape, stddev=0.1, mean=0.0), name="W")
                conv_a = self.conv(self.embedded_a_expand, w, [1, 1, 1, 1], "VALID")
                conv_b = self.conv(self.embedded_b_expand, w, [1, 1, 1, 1], "VALID")
                pooled_a = self.max_pool(conv_a, [1, sequence_length - filter_size + 1, 1, 1], [1, 1, 1, 1], 'VALID')
                pooled_b = self.max_pool(conv_b, [1, sequence_length - filter_size + 1, 1, 1], [1, 1, 1, 1], 'VALID')
                relu_a, relu_b = tf.nn.relu(pooled_a), tf.nn.relu(pooled_b)
                pooled_outputs_a.append(relu_a)
                pooled_outputs_b.append(relu_b)

        with tf.name_scope("output"):
            # 双向lstm，共享权重
            self.out1 = self.BiRNN(self.inputs_a, "a", "a", embedding_size, rnn_size, sequence_length, num_layers)
            self.out2 = self.BiRNN(self.inputs_b, "b", "b", embedding_size, rnn_size, sequence_length, num_layers)

        with tf.name_scope("result"):
            self.h_pool_a = tf.concat(pooled_outputs_a, 3)
            self.h_pool_b = tf.concat(pooled_outputs_b, 3)
            self.h_pool_flat_a = tf.squeeze(self.h_pool_a, axis=[1, 2])
            self.h_pool_flat_b = tf.squeeze(self.h_pool_b, axis=[1, 2])

        with tf.name_scope("loss"):
            self.feature_a = tf.concat((self.out1, self.h_pool_flat_a), axis=1)
            self.feature_b = tf.concat((self.out2, self.h_pool_flat_b), axis=1)
            self.distance = self.cosine_half_sita_square(self.feature_a, self.feature_b)
            print(self.distance)
            self.loss = self.contrastive_loss(self.label, self.distance, batch_size)

        with tf.name_scope("accuracy"):
            self.predict = 1 - tf.rint(self.distance)
            self.tmp_sim = tf.equal(tf.cast(self.predict, tf.int64), tf.cast(self.label, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(self.tmp_sim, tf.float32))

        print(self.feature_a)
        print(self.feature_b)
        print(self.distance)
        print(self.loss)
        print(self.predict)
        print(self.tmp_sim)
        print(self.accuracy)

    @staticmethod
    def conv(inputs, filter, strides, padding):
        return tf.nn.conv2d(input=inputs, filter=filter, strides=strides, padding=padding)

    @staticmethod
    def max_pool(value, ksize, strides, padding):
        return tf.nn.max_pool(value=value, ksize=ksize, strides=strides, padding=padding)

    @staticmethod
    def logistic_regression(x, w, b):
        return 1 / (1 + tf.exp(-1 * tf.nn.xw_plus_b(x, w, b)))

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
        return tf.sqrt((similarity + 1) / 2)

    @staticmethod
    def euclidean_distance_normalization(feature_a, feature_b):
        '''
        :param feature_a: 特征a, shape = [?, length]
        :param feature_b: 特征b, shape = [?, length]
        :return: 两个向量归一化后的欧式距离（或者是 1 - 相似度）
        '''
        euclidean_distance = tf.sqrt(tf.reduce_sum(tf.square(feature_a - feature_b), axis=1))
        euclidean_distance_norm = tf.div(euclidean_distance,
                                         tf.add(tf.sqrt(tf.reduce_sum(tf.square(feature_a), 1)),
                                                tf.sqrt(tf.reduce_sum(tf.square(feature_b), 1))))
        # euclidean_distance_norm = tf.reshape(euclidean_distance_norm, [-1])
        print("￥￥￥￥￥￥￥￥￥￥￥￥", euclidean_distance_norm)
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


# cnn = Model(sequence_length=35,
#             vocab_size=1000,
#             embedding_size=50,
#             filter_sizes=[1, 2, 3, 4, 5],
#             num_filters=20,
#             num_layers=1,
#             rnn_size=40,
#             batch_size=100)

