import tensorflow as tf
import pandas as pd
import datetime
import sys

from cnn import *
sys.path.append("../")
from PreProcess import *
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score


import matplotlib.pyplot as plt


class function(object):
    def __init__(self):
        pass

    @staticmethod
    def dev_data(x_test, y_test):
        test_a = [a for (a, b) in x_test]
        test_b = [b for (a, b) in x_test]
        y_test = np.squeeze(y_test, [1])
        return test_a, test_b, y_test

    @staticmethod
    def draw_chart(x, y, label, x_label, y_label, name):
        plt.plot(x, y, 'r', label=label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(name)
        plt.close()

    @staticmethod
    def test_result(sess, model, df, ans):
        vocab_model = pickle.load(open("../data/vocab.model", "rb"))
        batch_a = list(vocab_model.transform([data.text_to_wordlist(text) for text in df['question1'].values]))
        batch_b = list(vocab_model.transform([data.text_to_wordlist(text) for text in df['question2'].values]))

        batch_y = df['test_id'].values
        feed_dict = {
            model.input_sentence_a: batch_a,
            model.input_sentence_b: batch_b,
            model.label: batch_y,
            model.dropout_keep_prob: 0.5
        }

        answer = sess.run([model.ans], feed_dict=feed_dict)
        print(np.squeeze(np.array(answer), [0]).shape)
        return np.append(ans, answer)


class siamese_network_cnn(object):
    def __init__(self):
        # 定义CNN网络，对话窗口以及optimizer
        self.sess = tf.Session()
        self.cnn = Cnn(sequence_length=50,
                       vocab_size=73300,
                       embedding_size=128,
                       filter_sizes=[1, 2, 3, 4, 5, 6],
                       num_filters=100,
                       batch_size=1500)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cnn.loss, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())

        # 训练数据测试数据的获取，但是好像有点问题
        self.data_file = os.path.join(os.path.dirname(__file__), "../data/csv/train.csv")
        self.train_file = os.path.join(os.path.dirname(__file__), "../data/csv/train_train.csv")
        self.test_file = os.path.join(os.path.dirname(__file__), "../data/csv/train_test.csv")
        self.stop_words_file = "../data/stop_words_eng.txt"

        # self.data_create = data_create(self.data_file).get_one_hot()
        self.data_create = data(self.train_file, self.test_file, self.stop_words_file).get_one_hot()
        self.train_data, self.train_label = self.data_create.vec_train, self.data_create.label
        self.test_data, self.test_label = self.data_create.vec_test, self.data_create.test_label

        # 获取训练数据迭代器并且获取测试数据（用于神经网络验证）
        self.batches = data.get_batch(15, 1500, self.train_data, self.train_label)
        data_double, label_double = [], []
        for (d, l) in zip(self.test_data, self.test_label):
            if l == 0 and random.random() <= 0.5667:
                data_double.append(d)
                label_double.append(l)
        self.test_data = np.append(self.test_data, data_double, axis=0)
        self.test_label = np.append(self.test_label, label_double)
        self.test_a, self.test_b, self.y_test = function.dev_data(self.test_data, self.test_label)

        # 获取test数据
        self.df_test = pd.read_csv("../data/csv/test.csv")
        self.columns = ['question1', 'question2', 'test_id']
        self.df_test = self.df_test[self.columns].fillna(value="")

        # 定义数据变量，用来记录网络训练过程中的数据
        self.ans = []
        self.train_loss, self.train_accuracy = [], []
        self.test_loss, self.test_accuracy = [], []

        # tensorboard
        tf.summary.scalar("loss", self.cnn.loss)
        tf.summary.scalar("log_loss", self.cnn.log_loss)
        tf.summary.scalar("accuracy", self.cnn.accuracy)
        self.merged_summary_op_train = tf.summary.merge_all()
        self.merged_summary_op_test = tf.summary.merge_all()
        self.summary_writer_train = tf.summary.FileWriter("../summary/train", graph=self.sess.graph)
        self.summary_writer_test = tf.summary.FileWriter("../summary/test", graph=self.sess.graph)

    def train_step(self, a_batch, b_batch, label):
        '''
        神经网路的训练过程
        :param a_batch: Siamese网络左边的输入
        :param b_batch: Siamese网络右边的输入
        :param label: 标签
        :return: 训练网络，没有返回值
        '''
        feed_dict = {
            self.cnn.input_sentence_a: a_batch,
            self.cnn.input_sentence_b: b_batch,
            self.cnn.dropout_keep_prob: 0.2,
            self.cnn.label: label
        }
        _, summary, step, log_loss, loss, accuracy = self.sess.run(
            [self.optimizer, self.merged_summary_op_train, self.global_step, self.cnn.log_loss, self.cnn.loss, self.cnn.accuracy],
            feed_dict=feed_dict)
        self.summary_writer_train.add_summary(summary, step)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, log_loss {:g} accuracy {}".format(time_str, step, loss, log_loss, accuracy))
        self.train_loss.append(loss)
        self.train_accuracy.append(accuracy)

    def dev_step(self, a_batch, b_batch, label):
        '''
        神经网路的验证过程，查看网络是否收敛
        :param a_batch: Siamese网络左边的输入
        :param b_batch: Siamese网络右边的输入
        :param label: 标签
        :return: 验证网络，没有返回值
        '''
        feed_dict = {
            self.cnn.input_sentence_a: a_batch,
            self.cnn.input_sentence_b: b_batch,
            self.cnn.dropout_keep_prob: 1.0,
            self.cnn.label: label
        }
        log_loss, summary, step, acc, predict = self.sess.run(
            fetches=[self.cnn.log_loss, self.merged_summary_op_test, self.global_step, self.cnn.accuracy, self.cnn.predict],
            feed_dict=feed_dict
        )
        self.summary_writer_test.add_summary(summary, step)
        print(classification_report(y_true=label, y_pred=predict))
        print("\n**********\tlog_loss：", log_loss, "**********\n")
        self.test_loss.append(log_loss)
        self.test_accuracy.append(acc)

    def draw_chart(self):
        '''
        画出损失函数图，和正确率曲线图
        '''
        x = range(len(self.test_loss))
        function.draw_chart(
            x=x, y=self.test_loss, label="loss", x_label="batch", y_label="loss", name="./png/test_loss.png")
        function.draw_chart(
            x=x, y=self.test_accuracy, label="accuracy", x_label="batch", y_label="accuracy", name="./png/test_accuracy.png")

        x = range(len(self.train_loss))
        function.draw_chart(
            x=x, y=self.train_loss, label="loss", x_label="batch", y_label="loss", name="./png/train_loss.png")
        function.draw_chart(
            x=x, y=self.train_accuracy, label="accuracy", x_label="batch", y_label="accuracy", name="./png/train_accuracy.png")

    def predict(self):
        '''
        kaggle比赛的生成结果的过程
        '''
        start, length = 0, 10000
        while start < 2345796:
            stop = min(start + length, 2345796)
            index = pd.RangeIndex(start=start, stop=stop, step=1)
            print("start {}, end {}".format(start, stop))
            df = self.df_test.loc[index]
            self.ans = function.test_result(self.sess, self.cnn, df, self.ans)
            start += length

        pickle.dump(self.ans, open("../data/ans.pkl", "wb"))
        df = pd.DataFrame(data={"test_id": list(range(2345796)), "is_duplicate": self.ans},
                          columns=[["test_id", "is_duplicate"]])
        df.to_csv("../test.csv", index=False)

    def main(self, flag=False):
        '''神经网络的入口，整个网络的运行过程'''
        for batch in self.batches:
            x, y = zip(*batch)
            batch_a = np.array([a for (a, b) in x])
            batch_b = np.array([b for (a, b) in x])
            self.train_step(batch_a, batch_b, np.squeeze(y, axis=1))
            current_step = tf.train.global_step(self.sess, self.global_step)

            if current_step % 50 == 0:
                print("\nEvaluation:")
                self.dev_step(self.test_a, self.test_b, self.y_test)
                print("")
        self.draw_chart()

        # 将测试集拿出来训练
        if flag is True:
            self.batches = data.get_batch(7, 1500, zip(self.test_a, self.test_b), self.y_test)
            for batch in self.batches:
                x, y = zip(*batch)
                batch_a = np.array([a for (a, b) in x])
                batch_b = np.array([b for (a, b) in x])
                self.train_step(batch_a, batch_b, np.squeeze(y, axis=1))

        self.predict()
        print("Run the command line:\n"
              "--> tensorboard --logdir=summary"
              "\nThen open http://0.0.0.0:6006/ into your web browser")


if __name__ == '__main__':
    Net = siamese_network_cnn()
    Net.main()

