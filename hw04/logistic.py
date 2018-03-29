import numpy as np
import pandas as pd
import neural_net as nn

from scipy.io import arff
import sys

if __name__ == "__main__":
    # Input
    learning_rate = float(sys.argv[1])
    epochs = int(sys.argv[2])
    train_file_path = sys.argv[3]
    test_file_path = sys.argv[4]
    #learning_rate = 0.1
    #epochs = 10

    #train_file_path = "/Users/owner/Box Sync/UW/_cs760/hw04/diabetes_train.arff"
    #test_file_path = "/Users/owner/Box Sync/UW/_cs760/hw04/diabetes_test.arff"

    #train_file_path = "/Users/owner/Box Sync/UW/_cs760/hw04/magic_train.arff"
    #test_file_path = "/Users/owner/Box Sync/UW/_cs760/hw04/magic_test.arff"

    # Import data
    class_id = 'class'
    # Training set
    data_train, meta = arff.loadarff(train_file_path)

    meta_data = {}
    for i in meta.names():
        meta_data[i] = meta[i]
    meta_data = pd.DataFrame(meta_data)
    meta_data = meta_data[meta.names()]

    data_train = nn.normalize(pd.DataFrame(data_train), meta_data)
    data_train = nn.oneHot(data_train, meta_data)
    data_train = data_train.sample(frac=1).reset_index(drop=True)  # shuffle
    x_train = pd.DataFrame(
        data_train.iloc[:, data_train.columns.values != class_id])
    y_train = data_train[class_id]

    # Testing set
    data_test, _ = arff.loadarff(test_file_path)
    data_test = nn.normalize(pd.DataFrame(data_test), meta_data)
    data_test = nn.oneHot(data_test, meta_data)
    data_test = data_test.sample(frac=1).reset_index(drop=True)  # shuffle
    x_test = pd.DataFrame(
        data_test.iloc[:, data_test.columns.values != class_id])
    y_test = data_test[class_id]

    # NN description
    n_nodes_per_layer = [x_train.shape[1], 1]
    w = np.random.uniform(-0.01, 0.01,
                          nn.calculateNumOfW(n_nodes_per_layer))  # add bias

    net = nn.Neural(n_nodes_per_layer, w)

    # Training
    for i in range(1, epochs + 1):
        epoch_info = net.train(x_train, y_train, learning_rate)
        print("%d\t%.9f\t%d\t%d" %
              (i, epoch_info[0], epoch_info[1], epoch_info[2]))

    # Testing
    n_correct_class = 0
    y_pred = []
    for i in range(0, x_test.shape[0]):
        x = x_test.iloc[i, :]
        y = y_test[i]
        out, y_hat = net.predict(x)
        y_pred.append(y_hat)

        if y_hat == y:
            n_correct_class += 1
        print("%.9f\t%d\t%d" % (out, y_hat, y))
    print(str(n_correct_class) + "\t" + str(len(y_test) - n_correct_class))
    print("%.5f" % (nn.F1(y_test, y_pred)))
