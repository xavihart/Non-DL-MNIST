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
import skimage.feature as ft
from sklearn.ensemble import AdaBoostClassifier as adaboost
from sklearn.ensemble import BaggingClassifier

if __name__ == '__main__':

    C_list = [10]
    gamma_list = [0.001]
    start_time = time.time()
    sc = StandardScaler()
    x = np.load('./train.npy', allow_pickle=True)
    y = np.load('./text.npy', allow_pickle=True)
    print('data loaded successfully!' + '--'*5)
    train_x = x[:, 1:]
    train_y = x[:, 0]
    test_x = y[:, 1:]
    test_y = y[:, 0]
    train_x_hog = []
    test_x_hog = []
    for i in range(train_x.shape[0]):
        train_x_hog.append(ft.hog(train_x[i].reshape(28, 28), pixels_per_cell=(4, 4), cells_per_block=(4, 4)))
    for i in range(test_x.shape[0]):
        test_x_hog.append(ft.hog(test_x[i].reshape(28, 28), pixels_per_cell=(4, 4), cells_per_block=(4, 4)))
    train_x_hog = np.array(train_x_hog)
    test_x_hog = np.array(test_x_hog)
    train_x_hog = sc.fit_transform(train_x_hog)
    test_x_hog = sc.transform(test_x_hog)
    print("hog calculated successfully" + '--'*5)
    bagging_clf = BaggingClassifier(base_estimator=SVC(kernel='rbf', degree=2, gamma=0.001), max_samples=0.4)
    bagging_clf.fit(train_x_hog, train_y)
    print("bagging model fitted successfully!" + '--'*5)
    pred = bagging_clf.predict(test_x_hog)
    acc = (pred == test_y).sum() / test_y.shape[0]
    print("acc: {}".format(acc))
    f = open('./result/bagging/acc.txt', 'a')
    f.write("acc:" + str(acc) + "\n")
    f.close()
    s = pickle.dumps(bagging_clf)
    f = open('./result/bagging/bagging_model.model', 'wb+')
    f.write(s)
    f.close()
