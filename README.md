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
4. /data/lr_sentiment.model : logistics regression回归模型，用来预测情感正负性
5. /data/xgb_sentiment.model : xgboost回归模型，用来预测情感正负性
6. /data/


### 代码组织结构
<pre><code>
&nbsp;&nbsp;&nbsp;&nbsp;├── PreProcess.py 数据预处理
&nbsp;&nbsp;&nbsp;&nbsp;├── README.md
&nbsp;&nbsp;&nbsp;&nbsp;├── cnn_src
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── cnn.py:    cnn网络
&nbsp;&nbsp;&nbsp;&nbsp;│   └── train.py:    cnn网络的训练
&nbsp;&nbsp;&nbsp;&nbsp;├── data
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── csv
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── Tweets.csv:     推特数据（用来做情感分析）
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── new.csv.zip
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── test.csv：       test数据集
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── test.csv.zip：   test数据集的压缩包
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── test_part_aa：   test数据集分割的第一部分
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── test_part_ab：   test数据集分割的第二部分
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── train.csv：     训练数据集
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── train_test.csv： 分割训练数据集的测试数据，包括新特征
&nbsp;&nbsp;&nbsp;&nbsp;│   │   └── train_train.csv： 分割训练数据集的训练数据，包括新特征
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── feature.pkl：    特征
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── feature.pkl.zip
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── lr_sentiment.model：   情感分析的logistic regression模型
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── pkl
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── ans.pkl
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── bag.pkl
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── extra_feature.pkl：   训练数据集的extral feature（分成测试部分和训练部分，每个部分都是样本个数*17）
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── feature_old.pkl
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── test_distance.pkl：   测试集（需要提交到kaggle的测试数据集）的距离特征
&nbsp;&nbsp;&nbsp;&nbsp;│   │   ├── train.pkl
&nbsp;&nbsp;&nbsp;&nbsp;│   │   └── train_distance.pkl：   训练集的距离特征
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── stop_words_eng.txt
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── vocab.model
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── word_vec
&nbsp;&nbsp;&nbsp;&nbsp;│   │   └── xgb_sentiment.model
&nbsp;&nbsp;&nbsp;&nbsp;├── edit_distance.cpp
&nbsp;&nbsp;&nbsp;&nbsp;├── extral_features.py：提取extral feature
&nbsp;&nbsp;&nbsp;&nbsp;├── integration
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── integration.py：模型融合的代码（包含CNN和LSTM）
&nbsp;&nbsp;&nbsp;&nbsp;│   └── train.py：训练融合模型的代码
&nbsp;&nbsp;&nbsp;&nbsp;├── lstm_src
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── lstm.py
&nbsp;&nbsp;&nbsp;&nbsp;│   └── train.py
&nbsp;&nbsp;&nbsp;&nbsp;├── rnn_src
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── siamese_network.py
&nbsp;&nbsp;&nbsp;&nbsp;│   └── train.py
&nbsp;&nbsp;&nbsp;&nbsp;├── summary
&nbsp;&nbsp;&nbsp;&nbsp;├── test.py
&nbsp;&nbsp;&nbsp;&nbsp;└── 论文


extral_features.py：提取extral feature：
	class sentiment
	    :param twitter_path: 推特数据的路径
	    :param xgboost_path: xgboost模型情感分析的dump的路径和名称
	    :param lr_path： lr模型情感分析的dump的路径和名称
	
	class ManualFeatureExtraction
	    :param feature_path: 提取extra feature之后dump的路径（/data/feature.pkl）
	    :param data_file: 训练数据集：/data/csv/train.csv
	    :param lr_path: 逻辑回归模型的路径：/data/lr_sentiment.model
	    
	    :function tf_idf_word_match: 利用tf_idf值计算匹配程度
	    :function length_difference: 计算句子长度差值
	    :function LongCommonSequence: 句子的最长公共子序列
	    :function edit_distance_word: 句子之间的编辑距离
	    :function fuzzy_ratio: 计算句子之间的ratio
	    :function main: 计算句子的情感极性，并且综合前面的函数，计算出所有的数据，并且dump所有的手动提取的特征。
	    
	class distance
	    :param data_path: 数据
	    :param word2vecpath: Wordvec的路径
	    :param pkl: 距离特征的路径

PreProcess.py：预处理数据和生成新的数据：
	:function preprocess_tocsv: 统计数据并且生成相关图
	:function pre_split_train: 生成最后使用的数据
	                           包括train_test: 前五千条数据，用来测试
	                           同时包括train_train: 后面所有的数据，用来训练

	class data
		:param train_file_path: 训练数据文件路径
		:param test_file_path: 测试数据文件路径
		:param stop_words_file: 停用词文件
</code></pre>



#### Contrastive Loss (博客链接)
<pre><code>
http://blog.csdn.net/autocyz/article/details/53149760
</code></pre>


### 相关参考资料和论文

[1]  Ways of Asking and Replying in Duplicate Question Detection<br>
&ensp;&ensp;&ensp;&ensp;&ensp;http://www.aclweb.org/anthology/S17-1030 <br>

[2]  英文博客<br>
&ensp;&ensp;&ensp;&ensp;&ensp;https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07<br>

[3]  中文博客<br>
&ensp;&ensp;&ensp;&ensp;&ensp;https://www.leiphone.com/news/201802/X2NTBDXGARIUTWVs.html<br>

[4]  Quora Question Duplication <br>
&ensp;&ensp;&ensp;&ensp;&ensp;https://web.stanford.edu/class/cs224n/reports/2761178.pdf <br>

[5]  上海交通大学报告（非常重要）<br>
&ensp;&ensp;&ensp;&ensp;&ensp;http://xiuyuliang.cn/about/kaggle_report.pdf <br>

[6]  Deep text-pair classification with Quora’s 2017 question dataset<br>
&ensp;&ensp;&ensp;&ensp;&ensp;https://explosion.ai/blog/quora-deep-text-pair-classification <br>

[7]  NOTES FROM QUORA DUPLICATE QUESTION PAIRS FINDING KAGGLE COMPETITION <br>
&ensp;&ensp;&ensp;&ensp;&ensp;http://laknath.com/2017/09/12/notes-from-quora-duplicate-question-pairs-finding-kaggle-competition/ <br>