import itertools
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from attention import Attention
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRU, Flatten, Dropout
from deal_data_marking import get_data

X, y = get_data()

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
# 要统计的值
value_to_count = 0
# 使用条件表达式和np.count_nonzero统计值出现的次数
count = np.count_nonzero(y_test[:, 0] == value_to_count)
print(f"The value {value_to_count} appears {count} times in the array.")

# 要统计的值
value_to_count = 1
# 使用条件表达式和np.count_nonzero统计值出现的次数
count = np.count_nonzero(y_test[:, 0] == value_to_count)
print(f"The value {value_to_count} appears {count} times in the array.")

model = Sequential([
    GRU(40, activation='tanh', return_sequences=True),
    Dropout(0.2),
    Attention(),
    Flatten(),
    Dense(40, activation='relu'),
    Dropout(0.2),
    Dense(20, activation='relu'),
    Dropout(0.2),
    Dense(5)
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')  # 损失函数用均方误差
# 加载预训练权重
checkpoint_save_path = "./checkpoint/GRU.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=512, epochs=50, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
model.summary()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

################## predict ######################
# 测试集输入模型进行预测
start_time = time.time()
y_pre = model.predict(x_test)
end_time = time.time()
use_time = end_time-start_time

# pd.DataFrame(y_pre, columns=['normal', 'DoS']).to_csv('save_result/result.csv')
# 获取攻击分类结果
y_pre_attack = y_pre[:, 0:2]
y_real_attack = y_test[:, 0:2]
y_pre_attack = np.argmax(y_pre_attack, axis=1)
y_real_attack = np.argmax(y_real_attack, axis=1)

result = confusion_matrix(y_real_attack, y_pre_attack)


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.5f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confuse3.jpg', dpi=800)
    plt.show()


attack_types = ['0', '1']
plot_confusion_matrix(result, classes=attack_types, normalize=False, title='confusion matrix')

accuracy = accuracy_score(y_real_attack, y_pre_attack)
# 计算macro-average F1分数
f1_macro = f1_score(y_real_attack, y_pre_attack, average='macro')

# 计算精确度和召回率
# 'macro' 表示计算每个类别的指标，然后取平均值
precision_macro = precision_score(y_real_attack, y_pre_attack, average='macro')
recall_macro = recall_score(y_real_attack, y_pre_attack, average='macro')

# print("全局Accuracy:", accuracy)
print("Precision:", precision_macro)
print("Recall:", recall_macro)
print("F1 Score:", f1_macro)
print("全局Accuracy:", accuracy)
print('用时', use_time / y_test.shape[0])
# 绘制编码评估指标，将评估指标进行保存，方便评估
# 获取水印分类结果
y_pre_masking = y_pre[:, 2:]
y_real_masking = y_test[:, 2:]
df1 = pd.DataFrame(y_pre_masking, columns=['pre_a', 'pre_b', 'pre_c'])
df2 = pd.DataFrame(y_real_masking, columns=['real_a', 'real_b', 'real_c'])
pd.concat([df1,df2],axis=1).to_csv('result.csv')
# 创建一个新的图形
plt.figure()
# 设置一个颜色，例如红色
color = 'red'
# 循环遍历数组的每一列，并绘制它，但使用不同的线型
for i, column in enumerate(y_pre_masking.T):  # 使用.T转置数组以按列迭代
    linestyle = ['-', '--', ':'][i]  # 为每条线选择不同的线型
    plt.plot(column, label=f'Column {i + 1}', color=color, linestyle=linestyle)

# 设置一个颜色，例如红色
color = 'yellow'
# 循环遍历数组的每一列，并绘制它，但使用不同的线型
for i, column in enumerate(y_real_masking.T):  # 使用.T转置数组以按列迭代
    linestyle = ['-', '--', ':'][i]  # 为每条线选择不同的线型
    plt.plot(column, label=f'Column {i + 1}', color=color, linestyle=linestyle)
# 设置图例
plt.legend()

# 设置标题和坐标轴标签（可选）
plt.title('Real Label of Marking')
plt.xlabel('Index')
plt.ylabel('Value')

# 显示图形
plt.show()
# Precision: 1.0
# Recall: 1.0
# F1 Score: 1.0
# 全局Accuracy: 1.0

# 用时 0.0001220130937060543
