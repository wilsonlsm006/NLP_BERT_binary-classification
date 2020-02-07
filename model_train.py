
# func:根据用户搜索识别是否能打标传奇游戏标签
# 数据目录：./data_input
# input:train.csv  test.csv   训练数据集
# output:./bert_dump/XXX.hdf5 输出训练好的模型

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

# bear相关配置
maxlen = 510

##################################################################################
# 一些公共类的参数，基本上不会调整
##################################################################################
# bert相关配置路径
# 使用的词表
dict_path = "./bert_model/vocab.txt"
config_path = "./bert_model/bert_config.json"
checkpoint_path = "./bert_model/bert_model.ckpt"

# 标签类别数目
NCLASS = 2

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
func:将dataframe转化成np数组
输入：dataframe
输出：np数组
"""
def df_to_nparray(df):
    DATA_LIST = []
    for data_row in df.iloc[:].itertuples():
        DATA_LIST.append((data_row.ocr, to_categorical(data_row.label, NCLASS)))
    DATA_LIST = np.array(DATA_LIST)
    return DATA_LIST



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
func:计算模型在测试集上的评价指标
输入：dataframe,里面必须包含 label，prediction
输出：计算acc,precision,recall,fscore等
"""
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
def model_val(df):
    # 计算准确率
    acc = accuracy_score(df['label'], df['prediction'])
    # 计算精确率
    precision = metrics.precision_score(df['label'], df['prediction'])
    # 计算召回率
    recall  = metrics.recall_score(df['label'], df['prediction'])
    # 计算f得分
    fscore  = metrics.f1_score(df['label'], df['prediction'], average='weighted')
    # 计算混淆矩阵
    my_confusion_matrix = confusion_matrix(df['label'], df['prediction'])
    # 计算auc
    auc = metrics.roc_auc_score(df['label'], df['prediction'])#验证集上的auc值
    
    print("acc is %s,prediction is %s,recall is %s,f_score is %s,auc is %s" %(acc, precision, recall, fscore, auc))
    print(my_confusion_matrix)

"""
func:加载模型，并且用于预测新的数据
输入：需要预测的数据 dataframe, 模型
输出：预测的数据 np数组
"""
def model_load_predict(model_path, predict_df, result_path):

    # 需要将dataframe转化成 np数组格式
    DATA_LIST_TEST = []
    for data_row in predict_df.iloc[:].itertuples():
        DATA_LIST_TEST.append((data_row.ocr, to_categorical(0, NCLASS)))
    DATA_LIST_TEST = np.array(DATA_LIST_TEST)
    # 构建模型结构 nclass代表几分类模型
    my_model = build_bert(NCLASS)
    # 导入模型权重
    my_model.load_weights(model_path)
    # 这里需要得到数据类型都是元组，<class ‘tuple’>
    test_model_pred = np.zeros((len(DATA_LIST_TEST), NCLASS))

    # 模型预测
    test_D = data_generator(DATA_LIST_TEST, shuffle=False)
    test_model_pred = my_model.predict_generator(test_D.__iter__(), steps=len(test_D),verbose=1)

    # 模型预测的数据是 np数组的格式
    # 需要转化成df格式
    test_prediction = [np.argmax(x) for x in test_model_pred]
    
    test_predict_probability = [softmax(x)[1] for x in test_model_pred]
    
    predict_df['prediction'] = test_prediction
    
    predict_df['probability'] = test_predict_probability

    predict_df.to_csv(result_path, index=None)


if __name__ == '__main__':
    # 记录代码运行的开始时间
    train_start_time = datetime.datetime.now()

    ##################################################################################################
    #   这是模型训练需要配置的三个参数
    # input:
    # BERT_MODEL_NAME：使用的bert模型的目录
    # train_input_path:训练集输入目录
    # test_input_path:测试集集输入目录
    # NCLASS：本模型可以用于2分类，也可以用于多分类(大于2)
    # output:
    # model_save_path_finame 训练之后模型最终存储在哪里，由两部分组成模型存储目录和实验名称    
    ##################################################################################################
    
    # 模型输入数据：./data_input
    # train.csv 和 test.csv
    ROOT_PATH = Path("./") 
    DATA_ROOT = ROOT_PATH/"data_input" 
    
    # 输入输入格式是：ocr,label
    train_input_path = DATA_ROOT/"train.csv"
    test_input_path = DATA_ROOT/"test.csv"
    
    # 使用的bert模型目录
    BERT_MODEL_NAME = ROOT_PATH/"bert_model/"

    # 模型存储目录
    model_save_path = "./bert_dump/"
    # 可能一天做多个实验
    exp_name = "20200119_1602"
    model_save_path_finame = model_save_path+exp_name
    
    # 这里有个小trick
    # 通过控制是否实验标志，可以优先保证模型先跑起来
    TESTING=True
    
    print("导入bert模型路径：")
    print(BERT_MODEL_NAME)
    print("模型存储路径：")
    print(model_save_path_finame)
    print("训练集路径：")
    print(train_input_path)
    print("测试集路径：")
    print(test_input_path)
    
    ############################################################################################################
    # 模型训练
    # step1 划分训练集和测试集，获取训练数据 这里已经对训练数据进行了处理 比如NaN已经填充为0
    train_data_df = pd.read_csv(train_input_path)
    test_data_df = pd.read_csv(test_input_path)
    
    if TESTING:
        train_data_df = test_data_df.head(10240)
        test_data_df = test_data_df.head(1000)
    
    print("train_data_df is shape")
    print(train_data_df.shape)
    print("test_data_df is shape")
    print(test_data_df.shape)
    
    # 将dataframe转化成 numpy数组
    DATA_LIST_train = []
    for data_row in train_data_df.iloc[:].itertuples():
        DATA_LIST_train.append((data_row.ocr, to_categorical(data_row.label, NCLASS)))
    DATA_LIST_train = np.array(DATA_LIST_train)

    # step2 模型训练 这里已经训练好模型
    # 模型训练轮数
    epoch_num = 5
    # k折较差训练
    n_fold = 3
           
    # 模型训练获取最佳模型
    run_cv(n_fold, DATA_LIST_train, model_save_path_finame, epoch_num)
    
    # 记录代码的结束时间
    train_end_time = datetime.datetime.now()

    train_time_dur = (train_end_time - train_start_time).seconds

    # 计算耗时
    print("模型训练完成，模型训练耗时(分钟)：")
    print(starttime,endtime,train_time_dur/60)
    
