import numpy as np
import pandas as pd
import copy
from scipy.io import arff
import sys


# Class description
class Neural(object):
    def __init__(self,
                 n_nodes_per_layer=None,
                 w=None):
        self.n_nodes_per_layer = n_nodes_per_layer  # no bias
        self.n_layers = len(n_nodes_per_layer)
        self.layers = []

        left_bound = 0
        for l in range(0, self.n_layers - 1):
            tmp = (n_nodes_per_layer[l] + 1) * n_nodes_per_layer[l + 1]
            right_bound = left_bound + tmp

            w_tmp = w[left_bound: right_bound]
            x = [None for i in range(0, n_nodes_per_layer[l] + 1)]
            self.layers.append(Layer(x, w_tmp))
            left_bound = right_bound
        self.layers.append(Layer([None], [None]))  # output layer

    def forward(self, x):
        self.layers[0].updateX([1] + x)  # add bias

        for l in range(1, len(self.layers)):
            prev_x = self.layers[l - 1].getX()
            new_x = []

            if l != len(self.layers) - 1:  # hidden layer
                for i in range(0, self.layers[l].n_nodes - 1):
                    prev_w = self.layers[l - 1].getW(i)
                    new_x.append(sigmoid(sum(np.multiply(prev_x, prev_w))))

                # new_x = list((new_x - np.mean(new_x)) / np.std(new_x))  # norm
                new_x.insert(0, 1)  # add bias
            else:  # output layer
                prev_w = self.layers[l - 1].getW(0)  # 1 node - binary output
                new_x.append(sigmoid(sum(np.multiply(prev_x, prev_w))))

            self.layers[l].updateX(new_x)

    def backward(self, y, eta=1):
        # Update delta
        for l in reversed(range(1, self.n_layers)):
            if l == self.n_layers - 1:
                self.layers[l].nodes[0].delta = y - self.layers[l].nodes[0].x
            else:
                for i in range(1, self.layers[l].n_nodes):  # skip bias
                    node = self.layers[l].nodes[i]
                    next_deltas = [
                        node.delta for node in self.layers[l + 1].nodes]
                    if l != self.n_layers - 2:
                        next_deltas = next_deltas[1:]  # remove Bias

                    node.delta = node.x * (1 - node.x) * \
                        sum(np.multiply(node.w, next_deltas))

        # Update delta_w
        for l in reversed(range(0, self.n_layers - 1)):
            for i in range(0, self.layers[l].n_nodes):  # dont skip bias
                node = self.layers[l].nodes[i]
                # Decide if next layer has a bayes. No bias in output layer
                if l == self.n_layers - 2:
                    is_bias = 0
                else:
                    is_bias = 1

                node.delta_w = []
                for j in range(0, len(node.w)):
                    node.delta_w.append(
                        eta * self.layers[l + 1].nodes[j + is_bias].delta * node.x)

    def updateW(self):
        for l in range(0, len(self.layers) - 1):
            for node in self.layers[l].nodes:
                node.w = [w + delta_w for w, delta_w
                          in zip(node.w, node.delta_w)]

    def getPrediction(self):
        return(self.layers[self.n_layers - 1].nodes[0].x)

    def train(self, data_x, data_y, eta):
        n_correct_class = 0
        cross_entropy = 0
        for i in range(0, data_x.shape[0]):
            x = list(data_x.iloc[i, :])
            y = data_y[i]

            self.forward(x)
            self.backward(y, eta)
            self.updateW()
           # self.forward(x)
            out = self.getPrediction()

            y_pred = 1 * (out >= 0.5)
            if y_pred == y:
                n_correct_class += 1

            cross_entropy += -y * np.log(out) - (1 - y) * np.log(1 - out)

        return([cross_entropy, n_correct_class, len(data_y) - n_correct_class])

    def predict(self, x):
        # x - row, not a matrix
        if not isinstance(x, list):
            x = list(x)
        copy_nn = Neural.copyFrom(self)
        copy_nn.forward(x)
        out = copy_nn.getPrediction()
        y_pred = 1 * (out >= 0.5)
        return([out, y_pred])

    # Other constructors
    @classmethod
    def copyFrom(cls, nn):
        new_nn = copy.deepcopy(nn)
        return(new_nn)

    # Private instance
    def print(self):
        for layer in self.layers:
            print("hui")
            layer.print()


class Layer(object):
    # Every layer is described by set of nodes =>
    # have to know number of nodes and x, w for nodes
    # For every node we have 1 of x and len(w)/n_ndes of w
    # x with bias, w with bias
    def __init__(self, x, w):
        self.n_nodes = len(x)
        self.nodes = []
        n_w_per_node = int(len(w) / self.n_nodes)
        for i in range(0, self.n_nodes):
            w_tmp = w[i * n_w_per_node: (i + 1) * n_w_per_node]
            self.nodes.append(Node(x[i], w_tmp))

    def updateX(self, x):
        for i in range(0, self.n_nodes):
            self.nodes[i].x = x[i]

    def getX(self):
        x = [node.x for node in self.nodes]
        return(x)

    def getW(self, ind):
        w = [node.w[ind] for node in self.nodes]
        return(w)

    def normalize(self):
        x = self.getX()
        x_mean = np.mean(x)
        x_std = np.std(x)
        new_x = (x - x_mean) / x_std
        for i in range(0, len(new_x)):
            self.nodes[i].node.x = new_x[i]

    def print(self):
        for node in self.nodes:
            node.print()


class Node(object):
    def __init__(self, x=None, w=None):
        self.x = x
        self.w = w
        self.delta = None
        self.delta_w = None

    def print(self):
        print("w = " + str(self.w) + ", x = " + str(self.x) +
              ", delta = " + str(self.delta) + ", delta_w = " + str(self.delta_w))


# Help functions
def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res


def oneHot(data, meta, class_id="class"):
    nom_col = [x for x in meta.columns if meta[x][0] == "nominal"]
    new_data = {}
    new_data_colnames = []

    dict_id = 0
    for name in data.columns:
        if not name in nom_col:
            new_data[dict_id] = data[name]
            new_data_colnames.append(name)
            dict_id += 1
        else:
            range_val = meta[name][1]
            if name == class_id:
                new_data[dict_id] = pd.Series([range_val.index(x.decode())
                                               for x in data[name]])
                new_data_colnames.append(name)
                dict_id += 1
            else:
                for i in range(0, len(range_val)):
                    data_tmp = pd.Series(
                        [0 for i in range(0, len(data[name]))])
                    data_tmp.loc[data[name] == range_val[i].encode()] = 1
                    new_data[dict_id] = pd.Series(data_tmp)
                    new_data_colnames.append(name + "_" + range_val[i])
                    dict_id += 1

    new_data = pd.DataFrame(new_data)
    new_data.columns = new_data_colnames
    return(new_data)


def normalize(data, meta):
    numer_col = [x for x in meta.columns if meta[x][0] != "nominal"]
    if len(numer_col) > 0:
        numer_data = data[numer_col]
        mean = np.mean(numer_data)
        std = np.std(numer_data)
        norm_data = (numer_data - mean) / std
    else:
        return data

    # Replace values with normalised
    new_data = data
    for name in norm_data.columns:
        new_data[name] = norm_data[name]

    return(new_data)


def F1(y, y_pred):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred)

    ind_y = y == 1
    ind_y_pred = y_pred == 1

    TP = sum(ind_y & ind_y_pred)
    n_pos = sum(ind_y)
    n_pred_pos = sum(ind_y_pred)

    recall = TP / n_pos
    precision = TP / n_pred_pos

    return(2 * precision * recall / (precision + recall))


def calculateNumOfW(n_nodes_per_layer):
    # n_nodes_per_layer - not including bias
    n_w = 0

    for i in range(0, len(n_nodes_per_layer) - 1):
        n_w += (1 + n_nodes_per_layer[i]) * n_nodes_per_layer[i + 1]
    return(n_w)
