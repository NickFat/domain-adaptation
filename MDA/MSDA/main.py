import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import os
from tensorflow.keras.models import Sequential, Model, load_model
import datetime
from tensorflow.python.keras import backend as K
from keras.layers import Reshape,Input, Dense, ZeroPadding2D, Dropout, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.models import Model
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras import regularizers
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
import mmd
import model

@tf.function
def train_step(img_s1 ,img_s2 ,img_s3 ,img_t ,label_s1 ,label_s2 ,label_s3 ,G ,C1 ,CL1 ,C2 ,CL2 ,C3 ,CL3 ,opt_g ,opt_c1
               ,opt_cl1 ,opt_c2 ,opt_cl2 ,opt_c3 ,opt_cl3):
    with tf.GradientTape(persistent=True) as tape1:
        loss_mmd ,loss_cls ,loss_cen ,loss_di c =loss_all_domain(img_s1 ,img_s2 ,img_s3 ,img_t ,label_s1 ,label_s2
                                                                 ,label_s3 ,G ,C1 ,CL1 ,C2 ,CL2 ,C3 ,CL3)
        loss 1 =loss_mm d +loss_cl s +loss_ce n +loss_dic

    model_opt_c(loss1 ,tape1 ,C1 ,CL1 ,C2 ,CL2 ,C3 ,CL3 ,opt_g ,opt_c1 ,opt_cl1 ,opt_c2 ,opt_cl2 ,opt_c3 ,opt_cl3)
    # 参数添加
    loss_c1.update_state(loss_mmd)
    loss_c2.update_state(loss_cls)
    loss_c3.update_state(loss_cen)
    loss_c4.update_state(loss_dic)
    loss_total.update_state(loss1)

# 迭代训练
import time
EPOCHS = 200
BATCH_SIZE = 128
target_test_ac c =[]
train_los s =[]

for epoch in range(EPOCHS):
    star t =time.time()
    # 创建随机批次
    train_batc h =min(len(source1_train) ,len(source2_train) ,len(source3_train) ,len(target_train))
    indices = tf.random.shuffle(tf.range(train_batch))
    source1_train_shuffled = tf.gather(source1_train ,indices)
    source2_train_shuffled = tf.gather(source2_train ,indices)
    source3_train_shuffled = tf.gather(source3_train ,indices)
    target_train_shuffled = tf.gather(target_train ,indices)

    y1_train_shuffled = tf.gather(source1_label_train, indices)
    y2_train_shuffled = tf.gather(source2_label_train, indices)
    y3_train_shuffled = tf.gather(source3_label_train, indices)

    for batch in range(0, len(source1_train_shuffled), BATCH_SIZE):
        # 获取一个批次的输入和标签
        input1 = source1_train_shuffled[batch:batc h +BATCH_SIZE]
        input2 = source2_train_shuffled[batch:batc h +BATCH_SIZE]
        input3 = source3_train_shuffled[batch:batc h +BATCH_SIZE]
        input4 = target_train_shuffled[batch:batc h +BATCH_SIZE]


        labels1 = y1_train_shuffled[batch:batc h +BATCH_SIZE]
        labels2 = y2_train_shuffled[batch:batc h +BATCH_SIZE]
        labels3 = y3_train_shuffled[batch:batc h +BATCH_SIZE]

        input1 = tf.Variable(input1)
        input2 = tf.Variable(input2)
        input3 = tf.Variable(input3)
        input4 = tf.Variable(input4)

        labels1 = tf.Variable(labels1)
        labels2 = tf.Variable(labels2)
        labels3 = tf.Variable(labels3)

        # 训练一步
        train_step(input1 ,input2 ,input3 ,input4 ,labels1 ,labels2 ,labels3 ,G ,C1 ,CL1 ,C2 ,CL2 ,C3 ,CL3 ,opt_g
                   ,opt_c1 ,opt_cl1 ,opt_c2 ,opt_cl2 ,opt_c3 ,opt_cl3)


    # 输出训练进度
    template = 'Epoch {}, Loss_mmd: {}, Loss_cls:{}, Loss_cen:{}, Loss_dic:{}, Loss: {},'
    print(template.format(epoc h +1 ,loss_c1.result() ,loss_c2.result() ,loss_c3.result() ,loss_c3.result()
                          ,loss_total.result()))

    # 保存训练损失值
    train_loss.append(loss_total.result())


    # 重置指标
    loss_c1.reset_states()
    loss_c2.reset_states()
    loss_c3.reset_states()
    loss_c4.reset_states()
    loss_total.reset_states()

    # 测试
    feat1_t = G(target_test)
    temp1_t = C1(feat1_t)
    output1_t = CL1(temp1_t)

    feat2_t = G(target_test)
    temp2_t = C2(feat2_t)
    output2_t = CL2(temp2_t)

    feat3_t = G(target_test)
    temp3_t = C3(feat3_t)
    output3_t = CL3(temp3_t)

    output_ t =(output1_ t +output2_ t +output3_t ) /3

    pred t =[]
    true t =[]
    for i in range(0 ,output_t.shape[0]): a=output_t [i]
        predt.append(np.argmax(a))
    for j in range(0,target_l abel_test.shape[0]):
        b=target_ l abel_test[j]
        truet.append(np.argmax(b))
    # 计算准确率
    acc_t = accuracy_score(predt,truet)

    # 保存测试准确率
    target_test_acc.append(acc_t)

    end=time.ti m e()
    Time = end-start

    print('test_acc: {} ,Time: {}'.format(acc_t,Time))
    print('')