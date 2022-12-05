import csv
import pandas as pd
import numpy as np


def is_number(str):
    try:
        float(str)
        return True
    except ValueError:
        pass 
    return False

def statistic_with_file(file,lines):
    idx = 0
    data = []
    with open(file,'r',encoding='utf-8') as fp:
        reader=csv.reader(fp)
        for x in reader:
            if idx == 0:
                head = x
            elif idx in lines:
                for i in range(len(x)):
                    if is_number(x[i]):
                        x[i] = float(x[i])
                data.append(x)
            idx+=1
    assert len(data) == len(lines)
    data_df = pd.DataFrame(data, columns = head)
    print("=======Split:", int(data_df["split_type"][0]), "JK:", data_df["JK"][0], "Pooling: ", data_df["pooling"][0], "=======")
    print("Acc: ", round(data_df["acc"].mean(),4), "+/-", round(np.std(data_df["acc"])*100,2), "%")
    print("Precision:", round(data_df["precision"].mean(),4), "+/-", round(np.std(data_df["precision"])*100,2), "%")
    print("Recall:", round(data_df["recall"].mean(),4), "+/-", round(np.std(data_df["recall"])*100,2), "%")
    print("Score:", round(data_df["score"].mean(),4), "+/-", round(np.std(data_df["score"])*100,2), "%")


if __name__ == "__main__":
    files = ["result/max_[gcn].csv","result/mean_[gcn].csv","result/sum_[gcn].csv","result/max_[gcn].csv","result/mean_[gcn_split2].csv","result/sum_[gcn_split2].csv"]
    for file in files:
        statistic_with_file(file,[1,3,5])
        statistic_with_file(file,[2,4,6])