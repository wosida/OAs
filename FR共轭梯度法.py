import numpy as np
import matplotlib.pyplot as plt

# 目标函数 f(x) = x^2 + 5x + 6
def f(x):
    return x**2 + 5*x + 6

# 目标函数的梯度 f'(x) = 2x + 5
def grad_f(x):
    return 2*x + 5

# Armijo条件的线搜索
def armijo_line_search(x, d, alpha=1, beta=0.5, sigma=1e-4):
    t = alpha
    while f(x + t * d) > f(x) + sigma * t * np.dot(grad_f(x), d):
        t *= beta
    return t

# 非线性共轭梯度法（FR方法）
def nonlinear_conjugate_gradient(x0, max_iter=100, tol=1e-6):
    x = x0
    g = grad_f(x)  # 初始梯度
    d = -g          # 初始方向为负梯度方向
    func_vals = [f(x)]  # 用于记录每次迭代时的函数值
    for k in range(max_iter):
        # 如果梯度的模长小于容忍度，停止迭代
        if np.linalg.norm(g) < tol:
            print(f"Converged in {k} iterations")
            return x, func_vals
        # 线搜索，找到合适的步长
        t = armijo_line_search(x, d)
        # 更新x
        x = x + t * d
        # 计算新的梯度
        g_new = grad_f(x)
        # 使用Fletcher-Reeves公式更新共轭方向
        beta = np.dot(g_new, g_new) / np.dot(g, g)
        d = -g_new + beta * d  # 更新共轭方向
        g = g_new  # 更新梯度
        func_vals.append(f(x))  # 记录当前的函数值
    print("Maximum iterations reached")
    return x, func_vals

# 测试
x0 = np.array([10.0])  # 初始点
optimal_x, func_vals = nonlinear_conjugate_gradient(x0)
print(func_vals)

# 绘制函数值随迭代次数的下降图
plt.plot(func_vals, color='b')

plt.xlabel('Iteration')
plt.ylabel('Function value (log scale)')
plt.title('Function Value Decrease During Iterations')
plt.grid(True)
plt.show()

print(f"Optimal solution: x = {optimal_x}")
