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
   # data = np.genfromtxt('./MNIST/mnist_train.csv', delimiter=',')
   # data2 = np.genfromtxt('./MNIST/mnist_test.csv', delimiter=',')
   # np.save('train.npy', data)
   # np.save('text.npy', data2)

   # with open('./svm_model(simplified).bin', 'rb') as f:
  #      clf = pickle.load(f)
  #  print(clf)
    kernel = 'linear'
    #C_list = [0.01, 0.1, 1, 10, 100]
    #gamma_list = [0.0001, 0.001, 0.01, 0.1, 1]
    C_list = [10]
    gamma_list = [0.001]
    start_time = time.time()
    sc = StandardScaler()
    x = np.load('./train.npy', allow_pickle=True)
    y = np.load('./text.npy', allow_pickle=True)
    print(x.shape)
    print(y.shape)
    train_x = x[:, 1:]
    train_y = x[:, 0]
    test_x = y[:, 1:]
    test_y = y[:, 0]
    train_x /= 255.0
    test_x /= 255.0
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)
    with open('./result/rbf/optim_svmsize(2000).model', 'rb') as f:
        clf1 = pickle.load(f)
    #clf = SVC(kernel='rbf', C=5.0, gamma=0.0001)
    max_score = 0
    for C in C_list:
        for gamma in gamma_list:
            clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=2)
            clf.fit(train_x, train_y)
            score = clf.score(test_x, test_y)
            print("C:[", C, "]gamma:[", gamma, "]", "--score:", score)
            if score > max_score:
                max_score = score
                max_C = C
                max_gamma = gamma
                max_clf = clf

    print("best_C:", max_C, "best_gamma:", max_gamma, "score:", max_score)
    s = pickle.dumps((max_clf, sc))
    save_pth = os.path.join(os.getcwd(), "result","raw",  kernel)
    check_path(save_pth)
    f = open(os.path.join(save_pth, "optim_svm{}.model".format("size(60000)")), 'wb+')
    f.write(s)
    f.close()
    time_consumed = time.time() - start_time
    ans_test = clf.predict(test_x)
    acc = (ans_test == test_y).sum() / test_y.shape[0]
    print("acc", acc)
    print("time consumed:", time_consumed)
    print("over-----")
    f2 = open(os.path.join(save_pth, "acc.txt"), 'w')
    f2.write("train_set size:{}, test_set size:{}, acc in test set:{}".format(train_x.shape, test_x.shape, acc))
    f2.close()










