import numpy as np
import matplotlib.pyplot as plt


# 目标函数：f(x) = x^2 + 5x + 6
def objective_function(x):
    return x ** 2 + 5 * x + 6


# 目标函数的梯度：∇f(x) = 2x + 5
def gradient(x):
    return 2 * x + 5


# 目标函数的海森矩阵：H = 2
def hessian():
    return 2


# Cauchy步长的计算
def cauchy_step(grad, H, delta):
    # 计算Cauchy步长：步长必须小于信赖域
    p_C = -grad / H
    if np.linalg.norm(p_C) > delta:
        p_C = -delta * grad / np.linalg.norm(grad)
    return p_C


# Dogleg算法（引入信赖域）
def dogleg(H, x0, delta_init=1.0, max_iter=100, tol=1e-6):
    x = x0
    objective_values = [objective_function(x0)]  # 用于记录目标函数值
    delta = delta_init  # 初始信赖域大小

    for i in range(max_iter):
        grad = gradient(x)  # 计算梯度
        H_matrix = hessian()  # 计算海森矩阵

        # 计算Cauchy步长
        p_C = cauchy_step(grad, H_matrix, delta)

        # 计算Newton步长
        p_N = -grad / H_matrix

        # 计算Dogleg步长
        if np.linalg.norm(p_N) <= delta:
            # Newton步长小于信赖域半径，直接沿Newton方向
            p = p_N
        else:
            # 如果Newton步长大于信赖域半径，采用Dogleg路径
            p = p_C + (delta - np.linalg.norm(p_C)) / (np.linalg.norm(p_N) - np.linalg.norm(p_C)) * (p_N - p_C)

        # 更新x
        x_new = x + p
        objective_values.append(objective_function(x_new))

        # 输出当前迭代的信息
        print(f"Iteration {i + 1}: x = {x_new}, f(x) = {objective_function(x_new)}, Delta = {delta}")

        # 检查收敛条件
        if np.abs(x_new - x) < tol:
            print(f"Converged in {i + 1} iterations.")
            break

        # 更新信赖域大小
        if objective_function(x_new) < objective_function(x):
            # 如果目标函数改善了，扩大信赖域
            delta = min(2 * delta, 10)
        else:
            # 如果目标函数没有改善，缩小信赖域
            delta = delta / 2

        x = x_new

    print("Max iterations reached.")
    return x, objective_values


# 设置目标函数的测试
x0 = -5  # 初始点

# 执行Dogleg算法
result, objective_values = dogleg(hessian(), x0)

# 绘制目标函数值随迭代次数变化的图像
plt.plot(objective_values, 'ro-', markersize=5)
plt.title("Objective Function Value vs. Iteration with Trust Region")
plt.xlabel("Iteration")
plt.ylabel("Objective Function Value")
plt.xticks(np.arange(len(objective_values)))  # 确保x轴为整数
plt.grid(True)
plt.show()

print("Optimized x:", result)
print("Objective value at optimized x:", objective_function(result))
