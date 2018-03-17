import numpy as np
import pandas as pd


# Class description
class Node(object):
    def __init__(self, name=None, type=None, threshold=[],
                 parent=None, child=None,
                 n_per_label=[], label=None, sign=None):
        self.name = name
        self.type = type  # numeric of nominal
        self.threshold = threshold  # depending on type can be several
        self.sign = sign
        self.parent = parent
        self.child = child or []
        self.n_per_label = n_per_label
        self.label = label

    def printNode(self):
        print("name: ", self.name,
              "\n type: ", self.type,
              "\n threshold: ", self.threshold,
              "\n sign: ", self.sign,
              "\n parent: ", self.parent,
              "\n child: ", self.child,
              "\n n_per_label: ", self.n_per_label,
              "\n label: ", self.label)

    def printTree(self, n_spaces=0):
        if len(self.child) == 0:
            print(":", self.label)
        else:
            str_base = ""
            if self.parent != None:
                n_spaces += 1
                for i in range(0, n_spaces):
                    str_base += "|\t"

            for child in self.child:
                str_tree_branch = str_base + \
                    " ".join([child.name.lower(), child.sign])

                if isinstance(child.threshold, str):
                    str_tree_branch += " " + child.threshold
                else:
                    str_tree_branch += " {:.6f}".format(child.threshold)

                str_tree_branch += "".join([" [",
                                            str(child.n_per_label[0]),
                                            " ",
                                            str(child.n_per_label[1]),
                                            "]"])
                print(str_tree_branch, end="")
                if len(child.child) != 0:
                    print("")

                child.printTree(n_spaces)

    def predictSet(self, x):
        # Predicts set of y
        labels = []
        confs = []
        for i in range(0, len(x)):
            label, confidence = self.predict(x.iloc[i])
            labels.append(label)
            confs.append(confidence)
        return([labels, confs])

    def predict(self, x):
        # Predicts one instance
        if len(self.child) == 0:
            return([self.label,
                    (self.n_per_label[0] + 1) / (sum(self.n_per_label) + 2)])
        else:
            for child in self.child:
                attr_name = child.name
                threshold = child.threshold
                sign = child.sign
                if compareX(x[attr_name], threshold, sign) == True:
                    return(child.predict(x))

    def ROC(self, x, y):
        _, c = self.predictSet(x)
        c = np.array(c)
        ind = np.argsort(-np.array(c))

        c = c[ind]
        y = y[ind]

        n_pos = sum(y == "+")
        n_neg = sum(y == "-")
        TP = 0
        FP = 0
        last_TP = 0

        roc = []
        FPR = None
        TPR = None
        for i in range(0, len(y)):
            if i > 0 and c[i] != c[i - 1] and\
               y[i] == "-" and TP > last_TP:
                FPR = FP / n_neg
                TPR = TP / n_pos
                roc.append([FPR, TPR])
                last_TP = TP
            if y[i] == "+":
                TP += 1
            else:
                FP += 1
        if FPR != None:
            roc.append([FPR, TPR])
            roc = pd.DataFrame(roc, columns=["FPR", "TPR"])
        return(roc)


def growTree(x, y, meta, m=30, parent_node=None, y_labels=['+', '-']):
    node = Node(parent=parent_node,
                n_per_label=[sum(y == i) for i in y_labels])
    is_leaf = False

    # Stoping criterias
    if len(y) < m or len(set(y)) == 1:
        is_leaf = True

    cand_split_nodes = detCandidateSplit(x, meta)
    if len(cand_split_nodes) == 0:
        is_leaf = True

    best_split = findBestSplit(x, y, cand_split_nodes)
    if best_split == None:
        is_leaf = True

    # Grow a subtree
    if is_leaf == False:
        if best_split.type == "nominal":
            for threshold in best_split.threshold:
                data_ind = x[best_split.name] == threshold
                x_new = x[data_ind]
                y_new = y[data_ind]
                child_node = growTree(x_new, y_new, meta, m, node, y_labels)

                child_node.name = best_split.name
                child_node.type = best_split.type
                child_node.threshold = threshold
                child_node.sign = "="

                node.child.append(child_node)
        else:
            for i in range(0, 2):
                if i == 0:
                    data_ind = x[best_split.name] <= best_split.threshold
                    sign = "<="
                else:
                    data_ind = x[best_split.name] > best_split.threshold
                    sign = ">"
                x_new = x[data_ind]
                y_new = y[data_ind]
                child_node = growTree(x_new, y_new, meta, m, node, y_labels)

                child_node.name = best_split.name
                child_node.type = best_split.type
                child_node.threshold = best_split.threshold
                child_node.sign = sign

                node.child.append(child_node)
    else:
        node.name = 'class'
        if len(y) == 0 or node.n_per_label[0] == node.n_per_label[1]:
            node_tmp = node.parent
            while True:
                if node_tmp == None:
                    node.label = y_labels[0]
                    break
                if node_tmp.n_per_label[0] != node_tmp.n_per_label[1]:
                    node.label = y_labels[np.argmax(node_tmp.n_per_label)]
                    break
                else:
                    node_tmp = node_tmp.parent
        else:
            node.label = y_labels[np.argmax(node.n_per_label)]

    return(node)


def detCandidateSplit(data, meta):
    split_nodes = []
    for i in range(0, len(meta.columns)):
        attr_name = meta.columns.values[i]
        attr_type, attr_thresh = meta[attr_name]

        node_tmp = Node(attr_name, attr_type)
        if attr_type == "nominal":
            node_tmp.threshold = attr_thresh
        else:
            threshold_tmp = []
            sorted_train_set = list(data[attr_name].sort_values())

            for j in range(0, len(sorted_train_set) - 1):
                # Calculate attribute value midpoint.
                mid_point = (sorted_train_set[j] +
                             sorted_train_set[j + 1]) / 2.0
                threshold_tmp.append(mid_point)

            node_tmp.threshold = list(set(threshold_tmp))
            node_tmp.threshold.sort()  # set ruins order

        split_nodes.append(node_tmp)

    return(split_nodes)


def findBestSplit(x, y, cand_split_nodes):
    split_node = None
    info_gain = -1

    for node in cand_split_nodes:
        if node.type == "nominal":
            info_gain_tmp = infoGain(x[node.name], y)
            if info_gain_tmp > info_gain:
                split_node = node
                info_gain = info_gain_tmp
        else:
            for local_thresh in node.threshold:
                x_tmp = 0 * x[node.name]
                x_tmp[x[node.name] > local_thresh] = 1

                # Select a split based on information gain
                node_tmp = Node(node.name, node.type, local_thresh)
                info_gain_tmp = infoGain(x_tmp, y)

                if info_gain_tmp > info_gain:
                    split_node = node_tmp
                    info_gain = info_gain_tmp

                # In continious chose with lowest threshold
                if info_gain_tmp == info_gain and\
                   split_node.name == node_tmp.name and\
                   split_node.threshold > node_tmp.threshold:
                    split_node = node_tmp

    return(split_node)


def compareX(x, threshold, sign):
    if sign == "=":
        if x == threshold:
            return True
        else:
            return False
    if sign == "<=":
        if x <= threshold:
            return True
        else:
            return False
    if sign == ">":
        if x > threshold:
            return True
        else:
            return False


# General functions
def printPredictionSummary(y, y_predict):
    print("<Predictions for the Test Set Instances>")
    for i in range(0, len(y_predict)):
        print("%d: Actual: %s Predicted: %s" %
              (i + 1, y[i], y_predict[i]))
    print("Number of correctly classified: %d Total number of test instances: %d"
          % (sum(y == y_predict), len(y)))


def infoGain(x, y):
    return(calcEntropy(y) - calcCondEntropy(x, y))


def calcEntropy(y):
    uniq_values_y = set(y)
    H_y = 0
    for i in uniq_values_y:
        p_y = sum(y == i) / len(y)
        H_y += p_y * np.log2(p_y)

    return(-H_y)


def calcCondEntropy(x, y):
    uniq_values_x = set(x)
    H_y_given_x = 0
    for i in uniq_values_x:
        p_x = sum(x == i) / len(x)
        H_y_given_x += p_x * calcEntropy(y[x == i])
    return(H_y_given_x)


def decodeData(data, meta_type):
    ind = [i for i, x in enumerate(meta_type) if x == "nominal"]
    for i in ind:
        tmp = [x.decode() for x in data.iloc[:, i]]
        data.iloc[:, i] = tmp


if __name__ == "__main__":
    from scipy.io import arff
    import sys

    # Input
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    m = int(sys.argv[3])

    # train_file_path = "/Users/owner/Box Sync/UW/_cs760/hw02/credit_train.arff"
    # test_file_path = "/Users/owner/Box Sync/UW/_cs760/hw02/credit_test.arff"
    # m = 10

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
    decodeData(data_train, meta_data.iloc[0, :])

    x_train = data_train.iloc[:, 0:class_ind]
    y_train = data_train.iloc[:, class_ind]

    # Testing set
    data_test, _ = arff.loadarff(test_file_path)
    data_test = pd.DataFrame(data_test)
    decodeData(data_test, meta_data.iloc[0, :])

    x_test = data_test.iloc[:, 0:class_ind]
    y_test = data_test.iloc[:, class_ind]

    # Training
    tree = growTree(x_train, y_train,
                    meta_data.iloc[:, meta_data.columns != 'class'],
                    m=m, y_labels=meta_data.iloc[1, class_ind])
    tree.printTree()

    # Testing
    y_test_predict, _ = tree.predictSet(x_test)
    printPredictionSummary(y_test, y_test_predict)
