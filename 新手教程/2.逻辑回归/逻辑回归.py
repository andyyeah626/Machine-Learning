import numpy as np
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import mnist  # 使用 Keras 内置数据集

# 加载 MNIST 数据集（自动下载到 ~/.keras/datasets/）
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 将图像从 (28, 28) 展平为 (784,)，并归一化到 [0, 1]
X_train = X_train.reshape(-1, 784).astype('float32') / 255
X_test = X_test.reshape(-1, 784).astype('float32') / 255

# 训练逻辑回归模型
clf = LogisticRegression(penalty="l1", solver="saga", tol=0.1, max_iter=100)  # 增加 max_iter 避免收敛警告
clf.fit(X_train, y_train)

# 评估模型
score = clf.score(X_test, y_test)
print("Test score with L1 penalty: %.4f" % score)