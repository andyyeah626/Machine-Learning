import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = SVC(kernel="linear")  # 线性核
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', s=30)
ax = plt.gca()

# 获取当前坐标轴范围
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 生成网格以画决策边界
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 画决策边界与间隔
ax.contour(XX, YY, Z, colors='k',
           levels=[-1, 0, 1], alpha=0.7,
           linestyles=['--', '-', '--'])

# 画支持向量
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title("Linear SVM Decision Boundary")
plt.show()
