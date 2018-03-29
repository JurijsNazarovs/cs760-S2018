import numpy as np
import pandas as pd
import neural_net as nn
import importlib
importlib.reload(nn)

from scipy.io import arff
import sys

# NN description
w = [1, 2, -1, 2, 3, 1, 3, 1, -2, -2, 4, 0, 1, 1, 3, 1, 3, 2, 1]
x = [1, 3, 2, 1]
y = [1]

net = nn.Neural([4, 3, 1], w)
epoch_info = net.train(pd.DataFrame(x).transpose(), y, 1)
