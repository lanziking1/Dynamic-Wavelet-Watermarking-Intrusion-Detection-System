import numpy as np
import pandas as pd

def data_normalization(datelist):
    data = np.array(datelist)
    x_min = np.min(data)
    x_max = np.max(data)
    if x_max == x_min:
        return np.ones_like(data)
    return (data - x_min) / (x_max - x_min)


def get_data(time_step):
    db_data = []
    db_label = []

    #读取数据集文件
    Read_Data = pd.read_csv(r"  ")
    Data_Group = Read_Data.groupby("CAN ID")
    for name, group in Data_Group:
        group = group[['Timestamp', 'CAN ID', ' DLC',
                       'DATA0', 'DATA1', 'DATA2', 'DATA3',
                       'DATA4', 'DATA5', 'DATA6', 'DATA7',
                       'Label']].values

        for i in range(time_step, group.shape[0], 5):
            temp = group[i - time_step:i, 1:11]
            temp_data = []
            for j in range(temp.shape[1]):
                temp_data.append(data_normalization(temp[:, j]))

            sample = np.array(temp_data).T  # (time_step, 10)
            db_data.append(sample)

            db_label.append(
                1 if np.any(group[i - time_step:i, 11] == 1) else 0
            )

    db_data = np.array(db_data)
    db_label = np.array(db_label)

    np.random.seed(116)
    idx = np.random.permutation(len(db_data))
    db_data, db_label = db_data[idx], db_label[idx]

    db_data = np.expand_dims(db_data, axis=-1)  # (N, T, 10, 1)

    return db_data, db_label