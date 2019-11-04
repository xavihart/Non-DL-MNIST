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
sc = StandardScaler()
x = np.load('./train.npy', allow_pickle=True)
y = np.load('./text.npy', allow_pickle=True)
fig, ax = plt.subplots(6, 6, sharex=True, sharey=True)
train_x = x[:2000, 1:]
train_y = x[:2000, 0]
ax = ax.flatten()
ax[0].imshow(train_x[0].reshape(28, 28), cmap='gray_r')
for i, axis in enumerate(ax):
    label = str((train_y[i]).astype(np.int32))
    img = train_x[i].reshape(28, 28)
    axis.imshow(img, cmap='gray_r')
    axis.set_axis_off()
    axis.text(0.5, 1, "Number:" + label)
plt.savefig('./digits36.jpg')
plt.show()
print("finished drawing trainingset---")
with open('./result/rbf/optim_svmsize(5000).model', 'rb') as f:
    data = pickle.load(f)
clf1 = data[0]
sc = data[1]
_, ax2 = plt.subplots(6, 6)
ax_ = ax2.flatten()
test_x = y[:100, 1:]
test_y = y[:100, 0]
train_x /= 255.0
#train_x = sc.fit_transform(train_x)

for i, axis in enumerate(ax_):
    ans = test_y[i].astype(np.int32)
    img = test_x[i:i+1] / 255.0
    img = sc.transform(img)
    axis.imshow(test_x[i].reshape(28, 28), cmap='gray_r')
    pred = clf1.predict(img).astype(np.int32)
    axis.set_axis_off()
    if ans == pred:
        axis.text(0.75, 0.75, str(pred), color='b', horizontalalignment='right',
        verticalalignment='top')
        axis.text(0.75, 0.75,"correct", color='b', horizontalalignment='right',
        verticalalignment='bottom')
    else:
        axis.text(0.75, 0.75,str(pred), color='b', horizontalalignment='right',
        verticalalignment='top')
        axis.text(0.75, 0.75,"wrong", color='r', horizontalalignment='right',
        verticalalignment='bottom')
plt.savefig('./digit_test.jpg')
plt.show()