import pandas as pd
data_path = "/Users/owner/Box Sync/UW/_cs760/hw02/predictions.txt"

data = pd.read_csv(data_path)
data.columns = ["pred", "real"]
pred = list(set(data["pred"]))
real = list(set(data["real"]))

confus_matrix = []
for i in pred:
    conf_tmp = []
    for j in real:
        conf_tmp.append(sum((data.iloc[:, 0] == i) & (data.iloc[:, 1] == j)))

    confus_matrix.append(conf_tmp)

confus_matrix = pd.DataFrame(confus_matrix)
confus_matrix.columns = [str(x) for x in real]
confus_matrix.index = [str(x) for x in pred]
