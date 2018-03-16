import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Cnn(object):
    def __init__(self, sequence_length, vocab_size, embedding_size, filter_sizes, num_filters, batch_size):
        self.label = tf.placeholder(tf.float32, [None, ], name="label")
        self.outer_feature = tf.placeholder(tf.float32, [None, 17], name="outer_feature")
        self.input_sentence_a = tf.placeholder(tf.int32, [None, sequence_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.l2_loss = 0

        with tf.name_scope("embedding_layer"):
            self.W = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size],
                                                     dtype=tf.float32,
                                                     stddev=0.1,
                                                     mean=0.0), name="W")
            self.embedded_a = tf.nn.embedding_lookup(params=self.W, ids=self.input_sentence_a)
            self.embedded_b = tf.nn.embedding_lookup(params=self.W, ids=self.input_sentence_b)
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

        with tf.name_scope("result"):
            self.h_pool_a = tf.concat(pooled_outputs_a, 3)
            self.h_pool_b = tf.concat(pooled_outputs_b, 3)
            self.h_pool_flat_a = tf.squeeze(self.h_pool_a, axis=[1, 2])
            self.h_pool_flat_b = tf.squeeze(self.h_pool_b, axis=[1, 2])

            self.diff = self.h_pool_flat_a - self.h_pool_flat_b
            self.mul = tf.multiply(self.h_pool_flat_a, self.h_pool_flat_b)

            self.feature = tf.concat(
                values=[self.diff, self.mul, self.h_pool_flat_a, self.h_pool_flat_b, self.outer_feature],
                axis=1)
            self.feature_drop = tf.nn.dropout(self.feature, keep_prob=self.dropout_keep_prob)
            self.weight = tf.Variable(tf.truncated_normal(shape=[num_filters * len(filter_sizes) * 4 + 17, 1],
                                                          stddev=0.1,
                                                          mean=0.0))
            self.bias = tf.Variable(tf.truncated_normal(shape=[1], stddev=0.1, mean=0.0))
            self.result = tf.nn.xw_plus_b(self.feature_drop, self.weight, self.bias)
            self.logits = tf.nn.sigmoid(self.result)
            self.ans = tf.squeeze(self.logits, axis=1)

        with tf.name_scope("loss"):
            # log_loss
            # tf.nn.softmax_cross_entropy_with_logits:
            #   先计算logits的softmax函数值，如果Logits就是一个概率分布，会带来loss并不是实际loss的问题
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(self.weight) + tf.nn.l2_loss(self.bias)
            self.log_loss = tf.losses.log_loss(labels=self.label, predictions=self.ans)
            self.loss = self.log_loss# + 0.001 * self.l2_loss

        with tf.name_scope("accuracy"):
            self.predict = tf.rint(self.ans)
            self.tmp_sim = tf.equal(tf.cast(self.predict, tf.int64), tf.cast(self.label, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(self.tmp_sim, tf.float32))

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


# cnn = Cnn(sequence_length=35,
#           vocab_size=1000,
#           embedding_size=50,
#           filter_sizes=[1, 2, 3, 4, 5],
#           num_filters=20,
#           batch_size=101)

