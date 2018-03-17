import bayes_net as bn

from scipy.io import arff
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt

# Input
train_file_path = "/Users/owner/Box Sync/UW/_cs760/hw03/lymph_train.arff"
test_file_path = "/Users/owner/Box Sync/UW/_cs760/hw03/lymph_test.arff"

# Import data
class_id = 'class'
# Training set
data_train, meta = arff.loadarff(train_file_path)
class_ind = meta.names().index(class_id)

meta_data = {}
for i in meta.names():
    meta_data[i] = meta[i]
meta_data = pd.DataFrame(meta_data)
meta_data = meta_data[meta.names()]
# del meta

data_train = pd.DataFrame(data_train)
bn.decodeData(data_train, meta_data.iloc[0, :])

# Testing set
data_test, _ = arff.loadarff(test_file_path)
data_test = pd.DataFrame(data_test)
bn.decodeData(data_test, meta_data.iloc[0, :])

x_test = data_test.iloc[:, data_test.columns.values != class_id]
y_test = data_test[class_id]

# PR
tree_method = ["n", "t"]
pr_curve = {}
for method in tree_method:
    tree = bn.growTree(data_train, meta, root_name="class",  method=method)
    tree.sortChild(meta.names())
    tree.updatePostProb(data_train, meta)
    pr_curve[method] = tree.PR(x_test, y_test)

fig = plt.figure()
ax = fig.gca()
for key in pr_curve.keys():
    ax.plot(pr_curve.get(key)["recall"],
            pr_curve.get(key)["precision"], label=key)

plt.legend(loc='upper left')
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("PR")
# plt.show()
plt.savefig("pr.pdf")
