#!/bin/bash
# ***********************************************************************
# **  功能描述：用于脚本启动BERT模型
# **  创建者： 微信公众号：数据拾光者
# **  创建日期： 2020-02-22
# **  修改日期   修改人   修改内容
# ***********************************************************************

# 主目录
ROOT_PATH="./"

# 实验名称
LAB_FLAG='lsm_test_0224_2219'

# 数据存放目录
DATA_PATH=${ROOT_PATH}'/data_input'
# BERT预训练模型目录
BERT_MODEL_NAME=${ROOT_PATH}'/bert_model'
# 训练集路径
TRAIN_DATA=${DATA_PATH}'/train.csv'
# 模型存储路径
MODEL_SAVE_PATH=${ROOT_PATH}'/model_dump/'${LAB_FLAG}


python keras_train_classification.py --bert_model_name=${BERT_MODEL_NAME} \
    --train_data=${TRAIN_DATA} \
    --lab_flag=${LAB_FLAG} \
    --model_save_path=${MODEL_SAVE_PATH}
    
