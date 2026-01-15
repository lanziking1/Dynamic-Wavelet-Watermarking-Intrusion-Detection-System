import random
import pywt
import numpy as np
import pandas as pd

def Dynamic_Wavelet_Watermarking_Algorithm(signal, row, feature_label, wavelet='haar', l_max=6, l_target=3):
    best_level = 1
    min_diff = 3

    for l in range(1, l_max):
        try:
            coeffs = pywt.wavedec(signal, wavelet, level=l)
            cA = coeffs[0]  # approximation coefficients
            diff = abs(len(cA) - l_target)
            if diff < min_diff:
                min_diff = diff
                best_level = l
        except Exception as e:
            continue  # skip invalid decomposition
        if diff == 0:
            break

    # print("best_level", best_level)
    coeffs = pywt.wavedec(signal, wavelet, level=best_level)
    cA = coeffs[0]

    # Adjust cA length to l_target
    if len(cA) < l_target:
        # Interpolate to l_target
        cA_interp = np.interp(np.linspace(0, len(cA) - 1, l_target), np.arange(len(cA)), cA)
        cA = cA_interp
    elif len(cA) > l_target:
        # Truncate to first l_target elements
        cA = cA[:l_target]

    if len(cA) != l_target:
        raise ValueError("Adjusted cA length must be exactly l_target")

    Z = np.dot(cA, feature_label) / 3.0
    return np.float64(Z)


def embed_elements(data, elements_to_embed, pos1, flag=True):
    pos1 = random.randint(0, len(data) - 1)

    if not isinstance(elements_to_embed, list):
        elements_to_embed = [elements_to_embed]
    if pos1 == 0:
        new_data = elements_to_embed + data[:]
    elif pos1 == len(data) - 1:
        new_data = data[:pos1] + elements_to_embed + [data[-1]]
    else:
        new_data = data[:pos1] + elements_to_embed + data[pos1:]

    return new_data, pos1


def data_normalization(datelist):
    data = np.array(datelist)
    x_min = np.min(data)
    x_max = np.max(data)
    if x_max == x_min:
        normalized_data = np.ones_like(data)
    else:
        normalized_data = (data - x_min) / (x_max - x_min)

    return normalized_data.tolist()

def get_data(time_step):

    time_step = time_step
    feature_label = [1, 0.5, 0]     # Label of watermarking
    db_data = []
    db_y = []
    # Read the dataset
    DoS_Data = pd.read_csv(r"file")  # add file here
    DoS_Group = DoS_Data.groupby("CAN ID")
    for name, group in DoS_Group:
        group = group[['Timestamp', 'CAN ID', ' DLC', 'DATA0', 'DATA1', 'DATA2',
                       'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7', 'Label']].values
        # print(name)
        for i in range(time_step, group.shape[0] // 10):
            temp = group[i - time_step:i, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
            temp_data = []
            flag = True
            pos1 = None
            for j in range(temp.shape[1]):
                # Dynamic_Wavelet_Watermarking_Algorithm
                result = Dynamic_Wavelet_Watermarking_Algorithm(temp[:, j], row=j, feature_label=feature_label, wavelet='haar', l_max=4, l_target=3)
                new_data, pos1 = embed_elements(temp[:, j].tolist(), result, pos1=pos1, flag=flag)
                new_data = data_normalization(new_data)
                temp_data.append(new_data)
                flag = False

            db_data.append(np.array(temp_data).T)
            # Data label processing
            if np.any(group[i - time_step:i, [11]] == 1):
                db_y.append([1, 0, 1, 0.5, 0])
            else:
                db_y.append([0, 1, 1, 0.5, 0])


    db_data = np.array(db_data)
    db_y = np.array(db_y)
    # Data set shuffled
    np.random.seed(116)
    np.random.shuffle(db_data)
    np.random.seed(116)
    np.random.shuffle(db_y)

    db_data = np.reshape(db_data, (db_data.shape[0], time_step + 1, db_data.shape[2]))
    print("db_data.shape:", db_data.shape)

    return db_data, db_y



