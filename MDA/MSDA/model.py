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

def Feature():
    input_shape=(256,64,1)
    X_input=Input(input_shape)

    X=Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l1(0.001))(X_input)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2))(X)


    X=Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l1(0.001))(X)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2),strides=(2,2))(X)

    X=Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l1(0.001))(X)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2),strides=(2,2))(X)

    X=Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l1(0.001))(X)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2))(X)

    model=Model(inputs=X_input,outputs=X)

    return model


from keras.layers import LocallyConnected2D
def domain_class():
    inp=Feature().output
    X=Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same')(inp)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)


    X=Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same')(X)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)


    X=Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same')(X)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    X=MaxPooling2D((2,2))(X)


    X=LocallyConnected2D(filters=64,kernel_size=(1,1),strides=(1,1))(X)
    X=BatchNormalization()(X)
    X=Activation('relu',name='relu4')(X)
    X=MaxPooling2D((2,2))(X)


    X=Flatten()(X)

    X=Dense(64,activation='relu')(X)
    X=Dropout(0.3)(X)

    model=Model(inputs=inp,outputs=X)
    return model



def Classification():
    inp=domain_class().output
    X = Dense(5,activation='softmax')(inp)
    model=Model(inputs=inp,outputs=X)
    return model


G=Feature()
C1=domain_class()
CL1=Classification()

C2=domain_class()
CL2=Classification()

C3=domain_class()


# 定义损失函数和优化器
opt_g = tf.keras.optimizers.Adam(learning_rate=0.0001)

opt_c1 = tf.keras.optimizers.Adam(learning_rate=0.0001)
opt_cl1 = tf.keras.optimizers.Adam(learning_rate=0.0001)

opt_c2 = tf.keras.optimizers.Adam(learning_rate=0.0001)
opt_cl2 = tf.keras.optimizers.Adam(learning_rate=0.0001)

opt_c3 = tf.keras.optimizers.Adam(learning_rate=0.0001)
opt_cl3 = tf.keras.optimizers.Adam(learning_rate=0.0001)

# 定义评估指标
loss_c1 = tf.keras.metrics.Mean()
loss_c2 = tf.keras.metrics.Mean()
loss_c3 = tf.keras.metrics.Mean()
loss_c4 = tf.keras.metrics.Mean()
loss_total = tf.keras.metrics.Mean()

def euclidean(x1,x2):
    return tf.reduce_mean(tf.abs(x1-x2))

def center_loss(labels, features, alpha=0.6, num_classes=5):
    """
    获取center loss及更新样本的center
    :param labels: Tensor,表征样本label,非one-hot编码,shape应为(batch_size,).
    :param features: Tensor,表征样本特征,最后一个fc层的输出,shape应该为(batch_size, num_classes).
    :param alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
    :param num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    :return: Tensor, center-loss， shape因为(batch_size,)
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，如果labels已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)

    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features

    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    # 更新centers
    centers_update_op = tf.scatter_sub(centers, labels, diff)

    # 这里使用tf.control_dependencies更新centers
    with tf.control_dependencies([centers_update_op]):
        # 计算center-loss
        c_loss = tf.nn.l2_loss(features - centers_batch)

    return c_loss


def feat_all_domain(img_s1, img_s2, img_s3, img_t, G):
    return G(img_s1), G(img_s2), G(img_s3), G(img_t)


def feature_domain(feat1, feat2, feat3, feat_t, C1, C2, C3):
    s1 = C1(feat1)
    s2 = C2(feat2)
    s3 = C3(feat3)

    t1 = C1(feat_t)
    t2 = C2(feat_t)
    t3 = C3(feat_t)
    return s1, s2, s3, t1, t2, t3


def class_domain(s1, s2, s3, t1, t2, t3, CL1, CL2, CL3):
    imgs1 = CL1(s1)
    imgs2 = CL2(s2)
    imgs3 = CL3(s3)

    imgt1 = CL1(t1)
    imgt2 = CL2(t2)
    imgt3 = CL3(t3)

    return imgs1, imgs2, imgs3, imgt1, imgt2, imgt3


def mmd_loss(s1, s2, s3, t1, t2, t3):
    loss1 = MMD(s1, t1)
    loss2 = MMD(s2, t2)
    loss3 = MMD(s3, t3)
    return loss1, loss2, loss3


def softmax_loss_all_domain(imgs1, imgs2, imgs3, label_s1, label_s2, label_s3):
    criterion = tf.keras.losses.CategoricalCrossentropy()
    loss1 = criterion(label_s1, imgs1)
    loss2 = criterion(label_s2, imgs2)
    loss3 = criterion(label_s3, imgs3)

    return loss1, loss2, loss3


def centerL_loss(imgs1, imgs2, imgs3, label_s1, label_s2, label_s3):
    label_s1 = tf.cast(label_s1, tf.float32)
    label_s2 = tf.cast(label_s2, tf.float32)
    label_s3 = tf.cast(label_s3, tf.float32)

    #     label_s1 = tf.convert_to_tensor(label_s1, dtype='float32')
    #     label_s2 = tf.convert_to_tensor(label_s2, dtype='float32')
    #     label_s3 = tf.convert_to_tensor(label_s3, dtype='float32')

    loss1 = center_loss(K.argmax(label_s1, axis=-1), imgs1)
    loss2 = center_loss(K.argmax(label_s2, axis=-1), imgs2)
    loss3 = center_loss(K.argmax(label_s3, axis=-1), imgs3)

    #     loss1 = center1_loss(imgs1,label_s1,5,128, alpha=0.001)
    #     loss2 = center1_loss(imgs2,label_s2,5,128, alpha=0.001)
    #     loss3 = center1_loss(imgs3,label_s3,5,128, alpha=0.001)

    return loss1, loss2, loss3


def target_domain_class(imgt1, imgt2, imgt3):
    loss1 = euclidean(imgt1, imgt2)
    loss2 = euclidean(imgt1, imgt3)
    loss3 = euclidean(imgt2, imgt3)

    return loss1, loss2, loss3


def loss_all_domain(img_s1, img_s2, img_s3, img_t, label_s1, label_s2, label_s3, G, C1, CL1, C2, CL2, C3, CL3):
    feat_s1, feat_s2, feat_s3, feat_t = feat_all_domain(img_s1, img_s2, img_s3, img_t, G)

    s1, s2, s3, t1, t2, t3 = feature_domain(feat_s1, feat_s2, feat_s3, feat_t, C1, C2, C3)

    imgs1, imgs2, imgs3, imgt1, imgt2, imgt3 = class_domain(s1, s2, s3, t1, t2, t3, CL1, CL2, CL3)

    loss_mmd_1, loss_mmd_2, loss_mmd_3 = mmd_loss(s1, s2, s3, t1, t2, t3)

    loss_cls_1, loss_cls_2, loss_cls_3 = softmax_loss_all_domain(imgs1, imgs2, imgs3, label_s1, label_s2, label_s3)
    loss_cen_1, loss_cen_2, loss_cen_3 = centerL_loss(imgs1, imgs2, imgs3, label_s1, label_s2, label_s3)
    loss_dic_1, loss_dic_2, loss_dic_3 = target_domain_class(imgt1, imgt2, imgt3)

    loss_mmd = loss_mmd_1 + loss_mmd_2 + loss_mmd_3
    loss_cls = loss_cls_1 + loss_cls_2 + loss_cls_3
    loss_cen = loss_cen_1 + loss_cen_2 + loss_cen_3
    loss_dic = loss_dic_1 + loss_dic_2 + loss_dic_3

    return loss_mmd, loss_cls, loss_cen, loss_dic


def model_opt_c(loss, tape1, C1, CL1, C2, CL2, C3, CL3, opt_g, opt_c1, opt_cl1, opt_c2, opt_cl2, opt_c3, opt_cl3):
    # 计算梯度
    gradients = tape1.gradient(loss, G.trainable_variables)
    # 更新参数
    opt_g.apply_gradients(zip(gradients, G.trainable_variables))

    # 计算梯度
    gradients1 = tape1.gradient(loss, C1.trainable_variables)
    # 更新参数
    opt_c1.apply_gradients(zip(gradients1, C1.trainable_variables))

    # 计算梯度
    gradients12 = tape1.gradient(loss, CL1.trainable_variables)
    # 更新参数
    opt_cl1.apply_gradients(zip(gradients12, CL1.trainable_variables))

    # 计算梯度
    gradients2 = tape1.gradient(loss, C2.trainable_variables)
    # 更新参数
    opt_c2.apply_gradients(zip(gradients2, C2.trainable_variables))

    # 计算梯度
    gradients22 = tape1.gradient(loss, CL2.trainable_variables)
    # 更新参数
    opt_cl2.apply_gradients(zip(gradients22, CL2.trainable_variables))

    # 计算梯度
    gradients3 = tape1.gradient(loss, C3.trainable_variables)
    # 更新参数
    opt_c3.apply_gradients(zip(gradients3, C3.trainable_variables))

    # 计算梯度
    gradients32 = tape1.gradient(loss, CL3.trainable_variables)
    # 更新参数
    opt_cl3.apply_gradients(zip(gradients32, CL3.trainable_variables))



