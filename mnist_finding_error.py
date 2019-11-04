import numpy as np
import argparse
import pandas as pd
from sklearn.svm import LinearSVC, SVC, NuSVC
import os
import pickle
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
import time
import argparse
import os

def get_data(path):
    #data = pd.read_csv(path)
    data = np.genfromtxt(path, delimiter=',')
    x = data[:, 1:]
    x /= 255
    y = data[:, 0]
    return x, y

def prepare_svm(data_x, data_y, model='ovr'):
    clf = SVC(C=100.0, kernel='rbf', gamma=0.03, decision_function_shape='ovo')
    clf.fit(data_x, data_y)
    return clf

def get_explanation(clf, input_x, input_y) -> (int, float):
    # input is  L * (M)
    predict = clf.predict(input_x)
    acc = (predict == input_y).sum() / 100
    return predict, acc

def check_path(pth):
    if not os.path.exists(pth):
        os.mkdir(pth)
    if not os.path.exists(pth):
        os.makedirs(pth)

if __name__ == '__main__':
    with open('./result/raw/rbf/optim_svmsize(60000).model', 'rb') as f:
        clf = pickle.load(f)
    print("model loaded successfully !")
    sc = StandardScaler()
    x = np.load('./train.npy', allow_pickle=True)
    y = np.load('./text.npy', allow_pickle=True)
    print(x.shape)
    print(y.shape)
    train_x = x[:, 1:]
    train_y = x[:, 0]
    test_x = y[:, 1:]
    test_y = y[:, 0]
   # train_x /= 255.0
    #test_x /= 255.0
    train_x = sc.fit_transform(train_x)
    #test_x = sc.transform(test_x)
    print("Data loaded successfully!")



"""
for i, axis in enumerate(ax):
    axis.set_axis_off()
    axis.imshow(test_x[wrong_ans_list[i]].reshape(28, 28), cmap='gray_r')
    axis.text(0.5, 1, "prdc" + str(int(clf[0].predict(test_x[wrong_ans_list[i]].reshape(1, -1)))), color='r')
"""