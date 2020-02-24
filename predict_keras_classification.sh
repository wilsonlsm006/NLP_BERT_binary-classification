#!/bin/bash
# ***********************************************************************
# **  功能描述：用于脚本启动BERT模型
# **  创建者： 微信公众号：数据拾光者
# **  创建日期： 2020-02-22
# **  修改日期   修改人   修改内容
# ***********************************************************************

# 主目录
ROOT_PATH="./"


# 数据存放目录
DATA_PATH=${ROOT_PATH}'/data_input'
# BERT预训练模型目录
BERT_MODEL_NAME=${ROOT_PATH}'/bert_model'
# 训练集路径
TRAIN_DATA=${DATA_PATH}'/train.csv'
# 测试集路径
TEST_DATA=${DATA_PATH}'/test.csv'
# 测试集预测数据路径
TEST_PREDICT_DATA=${DATA_PATH}'/test_predict.csv'
# 手动选择导入哪个模型
MODEL_LOAD_PATH=${ROOT_PATH}'/model_dump/lsm_test_0224_2219_0.hdf5'


python keras_predict_classification.py --bert_model_name=${BERT_MODEL_NAME} \
    --test_data=${TEST_DATA} \
    --model_load_path=${MODEL_LOAD_PATH} \
    --test_predict_data=${TEST_PREDICT_DATA} 
