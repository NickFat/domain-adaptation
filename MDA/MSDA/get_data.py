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
import random

def get_data(path, file):
    mat = loadmat(os.path.join(path, file))
    data = pd.DataFrame(mat['emg1'])
    return data

def windowing(data1,data2):
    win_len = 256
    win_stride = 128
    data1 = pd.DataFrame(data1)
    data2 = pd.DataFrame(data2)
    idx = [i for i in range(win_len, len(data1), win_stride)]  # 得到每个滑动窗口的末端位置
    y = np.zeros([len(idx), ])
    X = np.zeros([len(idx), win_len, len(data1.columns)])  # 形成一个大小为len(idx)X400X64的三维空数组
    for i, end in enumerate(idx):  # 就是得到一个窗口的末端数值大小和这是第i个窗口
        start = end - win_len  # 每次用窗口的末端位置减去400就是初始位置，就可以得到窗口所在的坐标
        y[i] = data2.iloc[end,:]
        X[i] = data1.iloc[start:end, 0:64].values  # 取出这个400X12窗口内的数据，存入到前面定义的三维数组里面，就可以得到一个训练集
    # 取对应的动作类型标签，形成肌电图的标签
    # 取重复第几次的类型标签，形成肌电图对应重复次数的标签
    return X,y

