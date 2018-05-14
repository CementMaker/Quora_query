#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from tensorflow.contrib import rnn
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Model(object):
    def BiRNN(self, x, scope, embedding_size, rnn_size, sequence_length, num_layer, output_keep_prob):
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
        with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
                stm_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=output_keep_prob)
                stacked_rnn_fw.append(stm_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
                stm_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=output_keep_prob)
                stacked_rnn_bw.append(stm_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _, _ = rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell_m,
                                                         cell_bw=lstm_bw_cell_m,
                                                         inputs=x,
                                                         dtype=tf.float32,
                                                         scope=scope)
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

    def __init__(self,
                 sequence_length,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 num_layers,
                 rnn_size):
        self.label = tf.placeholder(tf.float32, [None, ], name="label")
        self.outer_feature = tf.placeholder(tf.float32, [None, 20], name="outer_feature")
        self.input_sentence_a = tf.placeholder(tf.int32, [None, sequence_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.lstm_keep_prob = tf.placeholder(tf.float32, name="lstm_keep_prob")

        with tf.name_scope("embedding_layer"):
            self.W = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size],
                                                     dtype=tf.float32,
                                                     stddev=0.1,
                                                     mean=0.0), name="W")
            self.embedded_a = tf.nn.embedding_lookup(params=self.W, ids=self.input_sentence_a)
            self.embedded_b = tf.nn.embedding_lookup(params=self.W, ids=self.input_sentence_b)

            # 循环神经网络的输入
            self.inputs_a = tf.unstack(self.embedded_a, axis=1)
            self.inputs_b = tf.unstack(self.embedded_b, axis=1)

            # CNN输入
            self.embedded_a_expand = tf.expand_dims(input=self.embedded_a, axis=-1)
            self.embedded_b_expand = tf.expand_dims(input=self.embedded_b, axis=-1)

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
            self.out1 = self.BiRNN(x=self.inputs_a,
                                   scope="a",
                                   embedding_size=embedding_size,
                                   rnn_size=rnn_size,
                                   sequence_length=sequence_length,
                                   num_layer=num_layers,
                                   output_keep_prob=self.lstm_keep_prob)
            self.out2 = self.BiRNN(x=self.inputs_b,
                                   scope="b",
                                   embedding_size=embedding_size,
                                   rnn_size=rnn_size,
                                   sequence_length=sequence_length,
                                   num_layer=num_layers,
                                   output_keep_prob=self.lstm_keep_prob)

        with tf.name_scope("result"):
            self.h_pool_a = tf.concat(pooled_outputs_a, 3)
            self.h_pool_b = tf.concat(pooled_outputs_b, 3)
            self.h_pool_flat_a = tf.squeeze(self.h_pool_a, axis=[1, 2])
            self.h_pool_flat_b = tf.squeeze(self.h_pool_b, axis=[1, 2])

            self.lstm_diff = self.out1 - self.out2
            self.lstm_mul = tf.multiply(self.out1, self.out2)
            self.cnn_diff = self.h_pool_flat_a - self.h_pool_flat_b
            self.cnn_mul = tf.multiply(self.h_pool_flat_a, self.h_pool_flat_b)

            self.feature = tf.concat(
                values=[self.cnn_diff, self.cnn_mul,
                        self.h_pool_flat_a, self.h_pool_flat_b, self.outer_feature,
                        self.lstm_diff, self.lstm_mul, self.out1, self.out2],
                axis=1
            )
            self.feature_drop = tf.nn.dropout(self.feature, keep_prob=self.dropout_keep_prob)
            self.weight = tf.Variable(tf.truncated_normal(
                shape=[rnn_size * 8 + num_filters * len(filter_sizes) * 4 + 20, 1],
                stddev=0.1,
                mean=0.0)
            )
            self.bias = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1, mean=0.0))
            self.result = tf.nn.xw_plus_b(self.feature, self.weight, self.bias)
            self.logits = tf.nn.sigmoid(self.result)
            self.ans = tf.squeeze(self.logits, axis=1)

        with tf.name_scope("loss"):
            # log_loss
            # tf.nn.softmax_cross_entropy_with_logits:
            #   先计算logits的softmax函数值，如果Logits就是一个概率分布，会带来loss并不是实际loss的问题

            keys = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.l2_loss = tf.contrib.layers.l2_regularizer(0.0002)
            self.l2_loss = tf.contrib.layers.apply_regularization(self.l2_loss, weights_list=keys)
            self.log_loss = tf.losses.log_loss(labels=self.label, predictions=self.ans)
            self.loss = self.log_loss + self.l2_loss

        with tf.name_scope("accuracy"):
            self.predict = tf.rint(self.ans)
            self.tmp_sim = tf.equal(tf.cast(self.predict, tf.int64), tf.cast(self.label, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(self.tmp_sim, tf.float32))

        values = [self.cnn_diff, self.cnn_mul,
                  self.h_pool_flat_a, self.h_pool_flat_b, self.outer_feature,
                  self.lstm_diff, self.lstm_mul, self.out1, self.out2]
        for index in values:
            print(index)

        print("*****************************************************")
        keys = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for key in keys:
            print(key.name)


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

#
# cnn = Model(sequence_length=35,
#             vocab_size=1000,
#             embedding_size=50,
#             filter_sizes=[1, 2, 3, 4, 5],
#             num_filters=20,
#             num_layers=1,
#             rnn_size=40)

