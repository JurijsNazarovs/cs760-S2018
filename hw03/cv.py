import bayes_net as bn

from scipy.io import arff
import pandas as pd
import random as rd
import numpy as np
from sklearn.model_selection import KFold

# Input
data_path = "/Users/owner/Box Sync/UW/_cs760/hw03/lymph_cv.arff"

# Import data
class_id = 'class'
data, meta = arff.loadarff(data_path)

meta_data = {}
for i in meta.names():
    meta_data[i] = meta[i]
meta_data = pd.DataFrame(meta_data)
meta_data = meta_data[meta.names()]

data = pd.DataFrame(data)
bn.decodeData(data, meta_data.iloc[0, :])

# CV
# Randomly shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

n_folds = 10
kf = KFold(n_splits=n_folds)
accuracies = {}
tree_method = ['n', 't']
for method in tree_method:
    accuracy = []
    for train, test in kf.split(data):
        # Train
        data_train = data.iloc[train, :]
        tree = bn.growTree(data_train, meta, root_name="class",  method=method)
        tree.sortChild(meta.names())
        tree.updatePostProb(data_train, meta)

        # Test
        data_test = data.iloc[test, :]
        x_test = pd.DataFrame(
            data_test.iloc[:,  data_test.columns.values != class_id])
        y_test = data_test[class_id]
        y_pred, _, _ = tree.predictSet(x_test)

        accuracy.append(np.mean(y_test == y_pred))
    accuracies[method] = accuracy

accuracies = pd.DataFrame(accuracies)

# t-test
t0 = 2.262157  # corresponds to 0.05/2
delta = accuracies.iloc[:, 0] - accuracies.iloc[:, 1]
delta_bar = np.mean(delta)
n = len(delta)
t = delta_bar / np.sqrt(1 / (n * (n - 1) * sum((delta - delta_bar)**2)))

if (t >= t0 or t <= -t0):
    print("H0 is rejected")
else:
    print("H0 is accepted")

# p-value is around 0.5 => cant reject
