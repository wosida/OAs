import numpy as np
import matplotlib.pyplot as plt


# 目标函数 f(x) = x^2 + 5x + 6
def f(x):
    return x ** 2 + 5 * x + 6


# 目标函数的梯度 f'(x) = 2x + 5
def grad_f(x):
    return 2 * x + 5


# Wolfe准则
def wolfe_condition(x, p, grad_f, alpha=1e-4, beta=0.9):
    # 计算 f(x + alpha * p)
    f_new = f(x + alpha * p)
    # 计算梯度内积
    grad_new = grad_f(x + alpha * p)
    if f_new <= f(x) + alpha * alpha * np.dot(grad_f(x), p) and np.dot(grad_new, p) >= beta * np.dot(grad_f(x), p):
        return True
    return False


# DFP方法
def dfp(f, grad_f, x0, max_iter=100, tol=1e-6):
    x = x0
    n = len(x0)
    H = np.eye(n)  # 初始Hessian的近似为单位矩阵
    iterations = []
    values = []

    for _ in range(max_iter):
        iterations.append(_)
        values.append(f(x))

        grad = grad_f(x)
        if np.linalg.norm(grad) < tol:
            break

        # 计算搜索方向
        p = -np.dot(H, grad)

        # 确定步长
        alpha = 1.0
        while not wolfe_condition(x, p, grad_f, alpha):
            alpha *= 0.5

        # 更新变量
        s = alpha * p
        x_new = x + s
        grad_new = grad_f(x_new)
        y = grad_new - grad

        # 更新Hessian的近似
        rho = 1.0 / np.dot(y, s)
        I = np.eye(n)
        H = np.dot((I - rho * np.outer(s, y)), np.dot(H, (I - rho * np.outer(y, s)))) + rho * np.outer(s, s)

        # 记录迭代过程中的函数值
        iterations.append(_)
        values.append(f(x))

        x = x_new

    return x, iterations, values


# 设置初始点
x0 = np.array([10.0])  # 初始值
# 运行DFP方法
optimal_x, iterations, values = dfp(f, grad_f, x0)
print(iterations)
print(values)

# 绘制函数值下降过程
plt.plot(iterations, values, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.title('Convergence of DFP Method')
plt.grid(True)
plt.show()

# 输出最优解
print(f"Optimal solution: x = {optimal_x}")
