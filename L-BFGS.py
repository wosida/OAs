import numpy as np
import matplotlib.pyplot as plt


# 定义目标函数 f(x) = x^2 + 5x + 6
def func(x):
    return x ** 2 + 5 * x + 6


# 目标函数的梯度 f'(x) = 2x + 5
def grad_func(x):
    return 2 * x + 5


# L-BFGS实现
def lbfgs(func, grad_func, x0, max_iter=50, m=10, tol=1e-5):
    # 初始化
    x = x0
    g = grad_func(x)  # 初始梯度
    history = [x]  # 用于记录每次迭代的x值
    s_list = []  # 存储位置差 s_k
    y_list = []  # 存储梯度差 y_k

    for k in range(max_iter):
        # 计算搜索方向 p_k
        p = -g  # 初始化搜索方向为负梯度
        p = np.array(p)  # 确保 p 是一个 NumPy 数组

        # 使用有限记忆L-BFGS更新
        if len(s_list) > 0:
            # 计算BFGS公式的方向
            q = p.copy()
            alpha = []
            for i in range(len(s_list) - 1, -1, -1):
                s, y = s_list[i], y_list[i]
                rho = 1.0 / np.dot(y, s)
                alpha_i = rho * np.dot(s, q)
                q -= alpha_i * y
                alpha.append(alpha_i)
            gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
            p = gamma * q
            for i in range(len(s_list) - 1, -1, -1):
                s, y = s_list[i], y_list[i]
                rho = 1.0 / np.dot(y, s)
                beta = rho * np.dot(y, p)
                p += s - beta * p

        # 线搜索（固定步长，简化处理）
        alpha = 1e-2
        x_new = x + alpha * p
        g_new = grad_func(x_new)

        # 判断收敛条件
        if np.linalg.norm(g_new) < tol:
            break

        # 更新位置和梯度
        s_k = x_new - x
        y_k = g_new - g

        # 存储s_k, y_k
        if len(s_list) >= m:
            s_list.pop(0)
            y_list.pop(0)
        s_list.append(s_k)
        y_list.append(y_k)

        x = x_new
        g = g_new
        history.append(x)  # 记录每次迭代的位置

    return x, history


# 测试函数
x0 = 0  # 初始点
x_opt, history = lbfgs(func, grad_func, x0)

# 绘制收敛曲线
x_values = np.linspace(-10, 10, 100)
y_values = func(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='Objective Function (f(x))', color='blue')
plt.scatter(history, [func(x) for x in history], color='red', label='Iterations', zorder=5)
plt.plot(history, [func(x) for x in history], color='red', linestyle='-', marker='o', markersize=5,
         label='Convergence Path')
plt.title("Manual L-BFGS Convergence on f(x) = x^2 + 5x + 6")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# 输出最终结果
print(f"Optimal x: {x_opt}")
