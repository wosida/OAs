#实现最小二乘拟合
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.normal(0, 1, 100)

# 定义目标函数
def f(x, w):
    return w[0] * x + w[1]

# 定义损失函数
def loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)

# 定义梯度
def grad(x, y, w):
    return np.array([-2 * np.mean((y - f(x, w)) * x), -2 * np.mean(y - f(x, w))])

# 梯度下降
def gradient_descent(x, y, w0, lr=0.01, max_iter=1000, tol=1e-6):
    w = w0
    for _ in range(max_iter):
        grad_w = grad(x, y, w)
        w_new = w - lr * grad_w
        if np.linalg.norm(w_new - w) < tol:
            break
        w = w_new
    return w

# 拟合
w0 = np.array([0.0, 0.0])
w_hat = gradient_descent(x, y, w0)

# 绘图
plt.scatter(x, y, label='data')
plt.plot(x, f(x, w_hat), color='red', label='fitting')
plt.legend()
plt.show()
