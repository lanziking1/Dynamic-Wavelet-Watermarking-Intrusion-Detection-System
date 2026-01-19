import numpy as np
import tensorflow as tf
import time
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_processing import get_data


# ================= 参数 =================
time_step = 12
EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-4


# ================= 获取数据 =================
X, y = get_data(time_step)

# 论文：预测模型仅用 Normal 训练
X_pred_train = X[y == 0]

# GNB：再划分训练 / 测试（非常关键）
X_all_train, X_all_test, y_train_cls, y_test_cls = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ================= ConvLSTM 预测模型 =================
model = models.Sequential([
    layers.ConvLSTM1D(
        filters=8,
        kernel_size=3,
        activation='tanh',
        padding='same',
        return_sequences=True,
        input_shape=(time_step, 10, 1)
    ),
    layers.ConvLSTM1D(
        filters=8,
        kernel_size=3,
        activation='tanh',
        padding='same'
    ),
    layers.Flatten(),
    layers.Dense(10)   # 预测特征（不是分类）
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss='mae'
)

model.fit(
    X_pred_train,
    X_pred_train[:, -1, :, :].reshape(len(X_pred_train), -1),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ================= Prediction Error =================
# ---- ConvLSTM 推理时间 ----
start_time = time.time()
y_pred = model.predict(X_all_test, verbose=0)
end_time = time.time()

conv_total_time = end_time - start_time
conv_avg_time = conv_total_time / len(X_all_test)

y_true = X_all_test[:, -1, :, :].reshape(len(X_all_test), -1)
errors_test = np.mean(np.abs(y_true - y_pred), axis=1)


# ---- 训练集误差（用于 GNB 拟合）----
y_pred_train = model.predict(X_all_train, verbose=0)
y_true_train = X_all_train[:, -1, :, :].reshape(len(X_all_train), -1)
errors_train = np.mean(np.abs(y_true_train - y_pred_train), axis=1)


# ================= GNB Classifier (论文 Section III-C) =================
def gnb_train(errors, labels):
    mu, sigma, prior = {}, {}, {}
    for c in [0, 1]:
        e = errors[labels == c]
        mu[c] = np.mean(e)
        sigma[c] = np.std(e) + 1e-6
        prior[c] = len(e) / len(errors)
    return mu, sigma, prior


def gnb_predict(errors, mu, sigma, prior):
    preds = []
    for e in errors:
        post = {}
        for c in [0, 1]:
            likelihood = (1 / (np.sqrt(2 * np.pi) * sigma[c])) * \
                         np.exp(-0.5 * ((e - mu[c]) / sigma[c]) ** 2)
            post[c] = likelihood * prior[c]
        preds.append(max(post, key=post.get))
    return np.array(preds)


# ---- 用训练集拟合 GNB（关键修正点）----
mu, sigma, prior = gnb_train(errors_train, y_train_cls)

# ---- GNB 推理时间 ----
start_time = time.time()
y_pred_cls = gnb_predict(errors_test, mu, sigma, prior)
end_time = time.time()

gnb_total_time = end_time - start_time
gnb_avg_time = gnb_total_time / len(errors_test)


# ================= Evaluation =================
acc = accuracy_score(y_test_cls, y_pred_cls)
prec = precision_score(y_test_cls, y_pred_cls)
rec = recall_score(y_test_cls, y_pred_cls)
f1 = f1_score(y_test_cls, y_pred_cls)

print("\n===== Detection Performance =====")
print(f"Accuracy : {acc:.6f}")
print(f"Precision: {prec:.6f}")
print(f"Recall   : {rec:.6f}")
print(f"F1-score : {f1:.6f}")

print("\n===== Runtime Performance =====")
print(f"ConvLSTM inference : {conv_avg_time * 1000:.6f} ms/sample")
print(f"GNB decision       : {gnb_avg_time * 1000:.6f} ms/sample")
print(f"Total IDS latency  : {(conv_avg_time + gnb_avg_time) * 1000:.6f} ms/sample")
