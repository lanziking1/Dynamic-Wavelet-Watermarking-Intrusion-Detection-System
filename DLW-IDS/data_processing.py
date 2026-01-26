import random
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# Extract features
def get_feature(signal, feature_label):
    spectral_flatness = np.std(signal) / np.abs(np.mean(signal)) if np.abs(np.mean(signal)) != 0 else 0
    # 计算均值、方差、偏度和峰度
    mean_value = np.mean(signal)
    variance = np.var(signal)
    skewness = skew(signal)
    kurtosis_value = kurtosis(signal,
                              fisher=False)

    use_signal = [spectral_flatness, mean_value, variance, skewness, kurtosis_value]
    result = convolve(signal=use_signal, kernel=feature_label)
    return result


# 利用卷积将label of marking 信息隐藏的嵌入到数据中
def convolve(signal, kernel, stride=2):
    result = []
    n = len(signal)
    k = len(kernel)

    for i in range(0, n - k + 1, stride):
        # 提取输入中的一部分来与卷积核相乘
        window = signal[i:i + k]
        # 计算卷积
        conv_value = sum(w * k for w, k in zip(window, kernel))
        # 添加到结果列表中
        result.append(conv_value)
    return result


# 进行位置嵌入，保证插入的水印信息，不相临
def embed_elements(data, elements_to_embed, flag=True, pos1=None, pos2=None):
    if flag == True:
        if len(data) < 3:
            raise ValueError("The data list must have at least 3 elements to embed [1, 0] non-adjacently.")

        pos1 = random.randint(0, len(data) - 2)
        if pos1 == 0:
            pos2_range = range(2, len(data) + 1)
        elif pos1 == len(data) - 2:
            pos2_range = range(0, pos1)
        else:
            pos2_range = list(range(pos1 + 2, len(data) + 1)) + list(range(0, pos1))

        pos2 = random.choice(pos2_range)

    if pos1 < pos2:
        new_data = data[:pos1] + [elements_to_embed[0].tolist()] + data[pos1:pos2] + [
            elements_to_embed[1].tolist()] + data[pos2:]
    else:
        new_data = data[:pos2] + [elements_to_embed[0].tolist()] + data[pos2:pos1] + [
            elements_to_embed[1].tolist()] + data[pos1:]
    return new_data, flag, pos1, pos2



def get_data():
    # 定义时间步，time_step
    time_step = 3
    # Label of marking
    feature_label = [1, 0, 1]
    # 保存所有数据和标签
    db_data = []
    db_y = []
    # 读取数据集
    DoS_Data = pd.read_csv("  ")
    DoS_Group = DoS_Data.groupby("CAN ID")
    for name, group in DoS_Group:
        group = group[['Timestamp', 'CAN ID', ' DLC', 'DATA0', 'DATA1', 'DATA2',
                       'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7', 'Label']].values
        for i in range(time_step, group.shape[0] % 100):
            # 按顺序获取每一列数据，获取输入数据
            temp = group[i - time_step:i, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            # 保存处理的样本，并进行转置
            temp_data = []
            # 保证每一个样本中插入的顺序一致
            flag = True
            pos1 = 0
            pos2 = 0
            for j in range(temp.shape[1]):
                result = get_feature(temp[:, j], feature_label=feature_label)
                new_data, flag, pos1, pos2 = embed_elements(temp[:, j].tolist(), result, flag=flag,
                                                            pos1=pos1, pos2=pos2)
                temp_data.append(new_data)
                print(new_data)
                flag = False
            # 将样本转置过后保存到db_data中
            db_data.append(np.array(temp_data).T)
            print(db_data[0])
            # 进行打标签，如果label为1，就是注入数据相反就是正常数据
            if np.any(group[i - time_step:i, [11]] == 1):
                db_y.append([1, 0, 1, 0, 1])
            else:
                db_y.append([0, 1, 1, 0, 1])

    # 数据集转化为numpy
    db_data = np.array(db_data)
    db_y = np.array(db_y)
    # 数据集打乱
    np.random.seed(116)
    np.random.shuffle(db_data)
    np.random.seed(116)
    np.random.shuffle(db_y)
    print("db_data.shape before reshape:", db_data.shape)
    db_data = np.reshape(db_data, (db_data.shape[0], time_step + 2, db_data.shape[2]))
    print("db_data.shape:", db_data.shape)
    return db_data, db_y



