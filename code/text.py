# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 16:58:26 2018

@author: sky
复现TextCNN
"""


import pandas as pd
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Embedding, LSTM, Dense, Activation
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

#读取数据
df_train = pd.read_csv(r'../data/train.csv',names=['id','content','label'], dtype={"content":str})
df_test = pd.read_csv(r'../data/test.csv', names=['id','content','label'], dtype={"content":str})
# 输出数据的维度
print("test shape is ",df_test.shape)
print('train shape is ',df_train.shape)
# 合并数据集
df = pd.concat([df_train,df_test],ignore_index=True)
df['content'] = df['content'].astype('str') #数据中部分行为整数

# 一些参数设置
# 最大词的数目
max_features =  30000
# 每句话的长度
maxlen = 100
# 批量的大小
batch_size = 32
# 词向量的维度
embedding_dims = 50
# 卷积核
filters = 250
# 核的大小
kernel_size = 3
# 隐藏层的维度
hidden_dims = 250
# 迭代次数
epochs = 8

train_length = df_train.shape[0]

# 文字转成数字表示
tokenizer = Tokenizer(num_words=max_features, filters='!！“”"#$%&（）()*+,，-.。/:：;<=>?？@[\]^_`{|}~\t\n、\r1234567890')
tokenizer.fit_on_texts(df['content'])
df['text'] = tokenizer.texts_to_sequences(df['content'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# 对齐数据，把数据维度变成maxlen的长度，其中，单词用数字表示
from keras.preprocessing.sequence import pad_sequences
df['text'] = list(pad_sequences(df['text'], maxlen=100))
# 切分数据集
x_train = df.ix[:train_length-1,'text'].tolist()
x_test = df.ix[train_length:,'text'].tolist()
x_train = np.array(x_train).reshape(-1,maxlen)
x_test = np.array(x_test).reshape(-1,maxlen)
print(x_train.shape, 'train sequences')
print(x_test.shape, 'test sequences')
#将label 做one-hot
y_train = to_categorical(df['label'][:df_train.shape[0]])
'''
df_result = pd.DataFrame(df_test['id'])
df_result['label'] = classes
df_result.to_csv('result.csv',header=None,index=False)
'''

print('Build model...')
# 初始化一个模型
model = Sequential()

# 添加词向量的层
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))

'''
#加载预训练word2vec模型
from gensim.models.word2vec import Word2Vec
sogou_w2v = Word2Vec.load(r'./sogou.model')
embeddings_matrix = np.zeros((max_features+1, 400))
for word, i in tokenizer.word_index.items():
    try:
        embedding_vector = sogou_w2v[word]
        embeddings_matrix[i] = embedding_vector
    except:
        pass
model.add(Embedding(max_features+1, 400, weights=[embeddings_matrix], input_length=maxlen, trainable=False))
'''

# dropout，控制过拟合
model.add(Dropout(0.5))

# 添加一维卷积
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# 最大化池化层
model.add(GlobalMaxPooling1D())

# 添加隐层
model.add(Dense(hidden_dims))
model.add(Dropout(0.5))
model.add(Activation('relu'))

# 最后一个16个节点的分类层
model.add(Dense(16))
model.add(Activation('softmax'))

#  编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# 输出模型的参数
print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,validation_split=0.2) #训练时间为若干个小时

# 预测测试集，并输出精度
classes = model.predict_classes(x_test)
#y_true = np.argmax(y_test,axis=1)
#acc = accuracy_score(y_true,classes)
#print("test acc is ",acc)
