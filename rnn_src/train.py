import datetime
import pickle

from PreProcess import *
from siamese_network import *

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

global_loss = []
global_accuracy = []
x_train, x_test, y_train, y_test = pickle.load(open("../data/train_test_query.pkl", "rb"))
test_a = [a for (a, b) in x_test]
test_b = [b for (a, b) in x_test]
y_test = np.squeeze(y_test, [1])
# y_test = np.squeeze([((y + 1) % 2, y) for y in y_test], [2])

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        lstm = siameseLSTM(sequence_length=30,
                           vocab_size=24000,
                           embedding_size=128,
                           hidden_units=50,
                           l2_reg_lambda=0.0,
                           batch_size=1500)

        # lstm = Model(num_layers=1,
        #              seq_length=30,
        #              embedding_size=150,
        #              vocab_size=74680,
        #              rnn_size=50,
        #              label_size=2,
        #              batch_size=1500)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(lstm.loss, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        def train_step(batch_a, batch_b, label):
            feed_dict = {
                lstm.input_x1: batch_a,
                lstm.input_x2: batch_b,
                lstm.input_y: label,
                lstm.dropout_keep_prob: 0.5
            }
            _, step, loss, accuracy = sess.run(
                [optimizer, global_step, lstm.loss, lstm.accuracy],
                feed_dict=feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, accuracy {}".format(time_str, step, loss, accuracy))
            global_loss.append(loss)
            global_accuracy.append(accuracy)

        def dev_step(batch_a, batch_b, label):
            feed_dict = {
                lstm.input_x1: batch_a,
                lstm.input_x2: batch_b,
                lstm.input_y: label,
                lstm.dropout_keep_prob: 0.5
            }
            step, loss, accuracy = sess.run([global_step, lstm.loss, lstm.accuracy], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, accuracy {}".format(time_str, step, loss, accuracy))

        batches = get_batch(60, 1500, x_train, y_train)
        for data in batches:
            x, y = zip(*data)
            batch_a = [a for (a, b) in x]
            batch_b = [b for (a, b) in x]

            y = np.squeeze(y, [1])
            train_step(batch_a, batch_b, y)
            current_step = tf.train.global_step(sess, global_step)

            if current_step % 300 == 0:
                print("\nEvaluation:")
                dev_step(test_a, test_b, y_test)
                print("")

        x = list(range(len(global_loss)))
        plt.plot(x, global_loss, 'r', label="loss")
        plt.xlabel("batches")
        plt.ylabel("loss")
        plt.savefig("loss_modify.png")
        plt.close()

        plt.plot(x, global_accuracy, 'b', label="accuracy")
        plt.xlabel("batches")
        plt.ylabel("accuracy")
        plt.savefig("accuracy.png")
        plt.close()

