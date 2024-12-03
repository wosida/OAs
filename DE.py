import random
import numpy as np

# 定义差分进化算法函数
def differential_evolution(objective_func, bounds, pop_size=100, F=0.8, CR=0.9, max_generations=200,dimension=2):
    # 初始化种群
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, dimension))
    
    # 进化过程
    for gen in range(max_generations):
        new_population = []
        for i in range(pop_size):
            x=population[i]
            indeies=list(range(pop_size))
            indeies.remove(i)
            # 随机选择三个不同的个体
            candidates = random.sample(indeies, 3)
            a, b, c = population[candidates]
            
            # 变异操作 DE/rand/1
            mutant = a + F * (b - c)
            
            # 交叉操作
            trial = np.array([mutant[j] if random.random() < CR else x[j] for j in range(dimension)])
            
            # 选择操作
            if objective_func(trial) < objective_func(x):
                new_population.append(trial)
            else:
                new_population.append(x)
        
        population = np.array(new_population)
    
    # 寻找最优解
    best_solution=min(population,key=objective_func)
    best_fitness = objective_func(best_solution)
    
    return best_solution, best_fitness

# 测试用的目标函数
def sphere_function(x):
    return sum(x**2)

# 定义搜索空间边界
bounds = (-10, 10)

# 调用差分进化算法进行优化
best_solution, best_fitness = differential_evolution(sphere_function, bounds)

print("Best solution:", best_solution)
print("Best fitness:", best_fitness)
