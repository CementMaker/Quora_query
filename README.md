### <center>Quora Question Pairs （短文本主题相似）</center>

### 使用Siamese网络结构：
1. 采用BLSTM最后一个神经元的输出，训练准确率９３，测试准确率为８３
   过拟合解决方法：期权，正则，但是还没有做．
   数据预处理还没有做完．
2. 单层LSTM有问题，可以继续搞一搞，但基本知道什么问题了
3.

### 数据（data文件夹）
1. /data/csv/train.csv : Quora公开的数据集，具有数据标签
2. /data/csv/test_part_aa, /data/csv/test_part_bb : 测试数据（test.py）split之后的数据，可以使用cat连接数据。
3. /data/vovab.model : VocabularyProcessor的模型（max_length = 60）

# Contrastive Loss (博客链接)
<pre><code>
http://blog.csdn.net/autocyz/article/details/53149760
</code></pre>


### 相关参考资料和论文

[1]  Ways of Asking and Replying in Duplicate Question Detection<br>
     &ensp;&ensp;&ensp;http://www.aclweb.org/anthology/S17-1030 <br>

[2]  英文博客<br>
     &ensp;&ensp;&ensp;https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07<br>

[3]  中文博客<br>
     &ensp;&ensp;&ensp;https://www.leiphone.com/news/201802/X2NTBDXGARIUTWVs.html<br>

[4]  Quora Question Duplication <br>
     &ensp;&ensp;&ensp;https://web.stanford.edu/class/cs224n/reports/2761178.pdf <br>

[5]  上海交通大学报告（非常重要）<br>
     &ensp;&ensp;&ensp;http://xiuyuliang.cn/about/kaggle_report.pdf <br>

[6]  Deep text-pair classification with Quora’s 2017 question dataset<br>
     &ensp;&ensp;&ensp;https://explosion.ai/blog/quora-deep-text-pair-classification <br>

[7]  NOTES FROM QUORA DUPLICATE QUESTION PAIRS FINDING KAGGLE COMPETITION <br>
     &ensp;&ensp;&ensp;http://laknath.com/2017/09/12/notes-from-quora-duplicate-question-pairs-finding-kaggle-competition/ <br>