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

def check_path(pth):
 if not os.path.exists(pth):
  os.mkdir(pth)
 if not os.path.exists(pth):
  os.makedirs(pth)
if __name__ == '__main__':
   # data = np.genfromtxt('./MNIST/mnist_train.csv', delimiter=',')
   # data2 = np.genfromtxt('./MNIST/mnist_test.csv', delimiter=',')
   #np.save('train.npy', data)
   #np.save('text.npy', data2)

   #with open('./svm_model(simplified).bin', 'rb') as f:
   #     clf = pickle.load(f)
   #print(clf)
    kernel = 'rbf'
    #C_list = [0.01, 0.1, 1, 10, 100]
    #gamma_list = [0.0001, 0.001, 0.01, 0.1, 1]
    C_list = [100]
    gamma_list = [0.1]
    start_time = time.time()
    sc = StandardScaler()
    x = np.load('./train.npy', allow_pickle=True)
    y = np.load('./text.npy', allow_pickle=True)
    print(x.shape)
    print(y.shape)
    train_x = x[:5000, 1:]
    train_y = x[:5000, 0]
    test_x = y[:500, 1:]
    test_y = y[:500, 0]
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

    print("feature size:", train_x_hog[0].shape)
    print("hog feature calculated successfully----")
    clf = SVC(kernel=kernel, degree=2, C=C_list[0], gamma=gamma_list[0])
    clf.fit(train_x_hog, train_y)
    acc = clf.score(test_x_hog, test_y)
    acc1 = clf.score(train_x_hog, train_y)
    print("acc:", acc)

"""
    saving_path = os.path.join("./result/hog", kernel)
    check_path(saving_path)
    s = pickle.dumps(clf)
    f = open(os.path.join(saving_path, "hog_svm.model"), 'wb+')
    f.write(s)
    f.close()
    print("model_saved ----")
    f2 = open(os.path.join(saving_path, "acc.txt"), 'a')
    f2.write("C:{}, gamma:{}train_set size:{}, test_set size:{}\naccuracy_train:{}, acc_test{}\n".format(C_list[0], gamma_list[0], train_x.shape, test_x.shape,acc1,  acc))
    f2.close()
"""