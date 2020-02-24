#! -*- coding:utf-8 -*-
import re, os, json, codecs, gc
import numpy as np
import pandas as pd
from random import choice
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from keras_bert import load_trained_model_from_checkpoint, Tokenizer

from pathlib import Path
from keras.layers import *
from keras.callbacks import *
from keras.models import Model, load_model
import keras.backend as K
from keras.optimizers import Adam
import keras_metrics
#import data_generator
from keras.utils import to_categorical
import datetime

from scipy.special import softmax
np.set_printoptions(precision=4)


# 将模型训练代码封装
# 通过外部传参应用到不同的任务中
import argparse
parser = argparse.ArgumentParser(description='binary_classification_train model')

# 需要传递的参数
parser.add_argument('--bert_model_name', type=str, default = None)
parser.add_argument('--train_data', type=str, default = None)
parser.add_argument('--model_save_path', type=str, default = None)
parser.add_argument('--lab_flag', type=str, default = None)

args = parser.parse_args()

BERT_MODEL_NAME = args.bert_model_name
LAB_FLAG = args.lab_flag
MODEL_SAVE_PATH = args.model_save_path
TRAIN_DATA_PATH = args.train_data

print("导入bert模型路径：")
print(BERT_MODEL_NAME)
print("模型存储路径：")
print(MODEL_SAVE_PATH)
print("训练集路径：")
print(TRAIN_DATA_PATH)

# bert相关配置路径
# 使用的词表
dict_path = BERT_MODEL_NAME+"/vocab.txt"
config_path = BERT_MODEL_NAME+"/bert_config.json"
checkpoint_path = BERT_MODEL_NAME+"/bert_model.ckpt"


# bear相关配置
maxlen = 510

# 模型训练参数
EPOCH_NUM = 5
# k折交叉训练
N_FOLD = 3
# 标签类别数目
NCLASS = 2

# 这里有个小trick
# 通过控制是否实验标志，可以优先保证模型先跑起来
TESTING=True

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class data_generator:
    def __init__(self, data, batch_size=8, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []


"""
func:交叉训练模型，获得最好的模型
输入：nfold(n折交叉训练),data(训练数据 np数组),data_label(数据标签，暂时无用),data_test(测试数据 np数组),model_save_path(模型存储路径),epoch_num(训练轮数)
输出：返回最佳模型
"""
# 训练模型
def run_cv(nfold, data, model_save_path, epoch_num):
    # 切分训练数据 切分成K折  这里切成成2部分
    kf = KFold(n_splits=nfold, shuffle=True, random_state=520).split(data)
    train_model_pred = np.zeros((len(data), 2))

    for i, (train_fold, test_fold) in enumerate(kf):
        X_train, X_valid, = data[train_fold, :], data[test_fold, :]

        # 训练集合验证集的数据量
        print(X_train.shape, X_valid.shape)

        model = build_bert(NCLASS)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)
        plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)
        # save_weights_only=False  保存模型所有数据(包括模型结构和配置信息)
        # save_best_only=True 只会保存最好模型
        model_save_final_str = model_save_path +"_"+str(i) + '.hdf5'
        print("model_save_final_str:")
        print(model_save_final_str)
        checkpoint = ModelCheckpoint(model_save_final_str, monitor='val_precision',
                                     verbose=2, save_best_only=True, mode='max', save_weights_only=True)

        train_D = data_generator(X_train, shuffle=True)
        valid_D = data_generator(X_valid, shuffle=True)

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=epoch_num,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[early_stopping, plateau, checkpoint],
        )

        #model.load_weights('./bert_dump/' + str(i) + '.hdf5')

        # return model
        train_model_pred[test_fold, :] = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)

        del model; gc.collect()
        K.clear_session()


"""
func:构建n分类的bert模型
输入:需要分成几类
输出：模型 
"""
def build_bert(nclass):
    #config_path = str(config_path)
    #checkpoint_path = str(checkpoint_path)

    # 读取bert预训练模型
    # keras_bert是在Keras下对Bert最好的封装是
    # 真正调用bert的就这么一行代码
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])

    # # 取出[CLS]对应的向量用来做分类
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(1e-5),
                  metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
    print(model.summary())
    return model

"""
func:模型训练函数
输入: train_data_path, model_save_path_finame
输出：训练完成得到的模型 
"""
def model_train(train_data_path, model_save_path_finame):
    # 获取训练数据集
    train_data_df = pd.read_csv(train_data_path)
    
    if TESTING:
        train_data_df = train_data_df.head(1024)
        
     # 将dataframe转化成 numpy数组
    DATA_LIST_train = []
    for data_row in train_data_df.iloc[:].itertuples():
        DATA_LIST_train.append((data_row.ocr, to_categorical(data_row.label, NCLASS)))
    DATA_LIST_train = np.array(DATA_LIST_train)

    # 模型训练轮数
    epoch_num = EPOCH_NUM
    # k折交叉训练
    n_fold = N_FOLD
    
    # 模型训练
    run_cv(n_fold, DATA_LIST_train, model_save_path_finame, epoch_num)
    
    
    
if __name__ == '__main__':
    # 记录代码运行的开始时间
    train_start_time = datetime.datetime.now()

    ############################################################################################################
    # 模型训练
    model_train(TRAIN_DATA_PATH, MODEL_SAVE_PATH)
    
    # 记录代码的结束时间
    train_end_time = datetime.datetime.now()

    train_time_dur = (train_end_time - train_start_time).seconds

    # 计算耗时
    print("模型训练完成，模型训练耗时(分钟)：")
    print(starttime,endtime,train_time_dur/60)
    
