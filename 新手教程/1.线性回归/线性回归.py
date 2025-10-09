import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def true_fun(X):
    return 1.5*X + 0.2

np.random.seed(0)
n_samples = 30

X_train = np.sort(np.random.rand(n_samples))
y_train = (true_fun(X_train) + np.random.randn(n_samples) * 0.05).reshape(n_samples, 1)

model = LinearRegression()
model.fit( X_train.reshape(-1, 1), y_train)

print("输出参数w：", model.coef_)
print("输出参数b：", model.intercept_)

X_test = np.linspace(0, 1, 100)
plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model")
plt.plot(X_test, true_fun(X_test), label="True function")
plt.scatter(X_train, y_train, edgecolor='b', s=20, label="Samples")
plt.legend()
plt.show()

