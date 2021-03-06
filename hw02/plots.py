#from dt_learn import growTree, decodeData, Node
import importlib
import dt_learn as dt
importlib.reload(dt)

from scipy.io import arff
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt

# Input
train_file_path = "/Users/owner/Box Sync/UW/_cs760/hw02/credit_train.arff"
test_file_path = "/Users/owner/Box Sync/UW/_cs760/hw02/credit_test.arff"
m = 10

# Import data
# Training set
data_train, meta = arff.loadarff(train_file_path)
class_ind = meta.names().index('class')

meta_data = {}
for i in meta.names():
    meta_data[i] = meta[i]
meta_data = pd.DataFrame(meta_data)
meta_data = meta_data[meta.names()]
#del meta

data_train = pd.DataFrame(data_train)
dt.decodeData(data_train, meta_data.iloc[0, :])

x_train = data_train.iloc[:, 0:class_ind]
y_train = data_train.iloc[:, class_ind]

# Testing set
data_test, _ = arff.loadarff(test_file_path)
data_test = pd.DataFrame(data_test)
dt.decodeData(data_test, meta_data.iloc[0, :])

x_test = data_test.iloc[:, 0:class_ind]
y_test = data_test.iloc[:, class_ind]


# Learning curve
# train_size = len(data_train)
# sample_sizes = [round(train_size * x) for x in [0.05, 0.1, 0.2, 0.5, 1]]
# sample_times = 10

# accuracy = []
# for size_tmp in sample_sizes:
#     accuracy_tmp = []
#     for j in range(0, sample_times):
#         ind = rd.sample(range(0, train_size), size_tmp)
#         tree = dt.growTree(x_train.iloc[ind], y_train.iloc[ind],
#                            meta_data.iloc[:, meta_data.columns != 'class'],
#                            m=m)
#         y_predict, _ = tree.predictSet(x_test)
#         accuracy_tmp.append(np.mean(y_test == y_predict))
#         if (size_tmp == train_size):
#             break
#     accuracy.append([min(accuracy_tmp), np.mean(
#         accuracy_tmp), max(accuracy_tmp)])

# accuracy = pd.DataFrame(accuracy, columns=["min", "avg", "max"])

# fig = plt.figure()
# ax = fig.gca()
# for i in range(0, len(accuracy)):
#     ax.plot(np.repeat(sample_sizes[i], accuracy.shape[1]),
#             accuracy.iloc[i], color="black")
# ax.plot(sample_sizes, accuracy.iloc[:, 1], color="black")
# plt.xlabel("sample_size")
# plt.ylabel("accuracy")
# plt.title("learning_curve")
# # plt.show()
# plt.savefig("learning_curve.pdf")


# ROC
tree = dt.growTree(x_train, y_train,
                   meta_data.iloc[:, meta_data.columns != 'class'],
                   m=m)
roc = tree.ROC(x_test, y_test)

fig = plt.figure()
ax = fig.gca()
ax.plot(roc["FPR"], roc["TPR"], color="black")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
# plt.show()
plt.savefig("roc.pdf")
