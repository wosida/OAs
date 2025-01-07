import numpy as np
import matplotlib.pyplot as plt


# 目标函数 f(x) 和其梯度
def f(x):
    return x ** 2 + 5 * x + 6


def grad_f(x):
    return 2 * x + 5


# 牛顿法的海森矩阵（对于简单的二次函数，海森矩阵是常数）
def hessian(x):
    return 2  # 对于 f(x) = x^2 + 5x + 6，海森矩阵是常数 2


# Wolfe条件中的参数
alpha = 0.1  # 步长因子
beta = 0.7  # 步长因子
gamma = 0.9  # 曲率条件参数


# 牛顿法
def newton_method(x0, max_iter=100):
    x = x0
    iter_values = []  # 记录每次迭代的目标函数值
    grad0 = grad_f(x)

    for k in range(max_iter):
        iter_values.append(f(x))
        # 计算当前梯度
        grad = grad_f(x)

        # 计算牛顿方向：d_k = -H^{-1} * grad
        H = hessian(x)  # 计算海森矩阵
        d_k = -grad / H  # 对于简单的二次函数，H是常数，因此 d_k 直接是梯度除以H

        # 线搜索：通过Wolfe条件计算步长alpha_k
        # 在这里加入曲率条件的检查
        alpha_k = 1.0
        while True:
            # 计算步长后的目标函数值
            f_new = f(x + alpha_k * d_k)
            f_curr = f(x)
            grad_new = grad_f(x + alpha_k * d_k)

            # 充分下降条件
            sufficient_descent_condition = f_new <= f_curr + alpha * alpha_k * np.dot(grad, d_k)

            # 曲率条件
            curvature_condition = np.dot(grad_new, d_k) >= gamma * np.dot(grad, d_k)

            # 如果满足两个条件，则退出循环
            if sufficient_descent_condition and curvature_condition:
                break
            else:
                alpha_k *= beta  # 如果条件不满足，缩小步长

        # 更新 x
        x = x + alpha_k * d_k

        # 记录当前的函数值
        iter_values.append(f(x))

        # 打印每次迭代的值（可选）
        print(f"Iteration {k + 1}: x = {x}, f(x) = {f(x)}")

    return iter_values


# 初始化参数并运行牛顿法
x0 = 10  # 初始值
max_iter = 20  # 最大迭代次数
iter_values = newton_method(x0, max_iter)

# 绘制迭代次数 vs 函数值的下降图
plt.plot(iter_values)
plt.xlabel('Iteration')
plt.ylabel('Function Value f(x)')
plt.title('Newton Method: Function Value vs Iteration')
plt.grid(True)
plt.show()
