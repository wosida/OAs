import random

# 适应度函数（最小化问题）
def fitness_function(x):
    return x**2

# 创建种群
def create_pop(pop_size, chromosome_length):
    pop = []
    for _ in range(pop_size):
        individual = [random.randint(0, 1) for _ in range(chromosome_length)]
        pop.append(individual)
    return pop

# 计算适应度
def calculate_fitness(individual):
    x = int(''.join(map(str, individual)), 2)
    return fitness_function(x)

# 选择操作（轮盘赌选择）
def selection(pop, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_individual=random.choices(pop, weights=probabilities)[0]
    return selected_individual

# 交叉操作
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

# 变异操作
def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

# 遗传算法主程序
def genetic_algorithm(pop_size, chromosome_length, generations, crossover_rate, mutation_rate):
    pop = create_pop(pop_size, chromosome_length)
    
    for generation in range(generations):
        new_pop=[]
        # 保留最优个体（精英策略）
        elite = min(pop, key=calculate_fitness)
        new_pop.append(elite)

        fitness_values = [calculate_fitness(individual) for individual in pop]
        
        while len(new_pop) < pop_size:
            parent1 = selection(pop, fitness_values)
            parent2 = selection(pop, fitness_values)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_pop.extend([child1, child2])
        
        pop = new_pop
    
    #best_individual = min(pop, key=lambda x: calculate_fitness(x)) Python标准库中并没有直接提供argmin函数，而是通过min()函数结合key参数来实现类似的功能
    best_individual = min(pop, key=calculate_fitness)
    best_x = int(''.join(map(str, best_individual)), 2)  #map返回可迭代对象
    
    return best_x

# 设置参数并运行遗传算法
pop_size = 50
chromosome_length = 8
generations = 100
crossover_rate = 0.5
mutation_rate = 0.5
best_solution = genetic_algorithm(pop_size, chromosome_length, generations, crossover_rate, mutation_rate)

print("最优解为:", best_solution)
#print(''.join(['a','a']))