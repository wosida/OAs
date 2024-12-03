import random

# 适应度函数（最小化问题）
def fitness_function(x):
    return x**2

# 创建种群
def create_population(population_size, chromosome_length):
    population = []
    for _ in range(population_size):
        individual = [random.randint(0, 1) for _ in range(chromosome_length)]
        population.append(individual)
    return population

# 计算适应度
def calculate_fitness(individual):
    x = int(''.join(map(str, individual)), 2)
    return fitness_function(x)

# 锦标赛选择
def tournament_selection(population, tournament_size):
    selected_individuals = random.sample(population, tournament_size)
    selected_individuals.sort(key=calculate_fitness)
    return selected_individuals[0]

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
def genetic_algorithm(population_size, chromosome_length, generations, tournament_size, crossover_rate, mutation_rate):
    population = create_population(population_size, chromosome_length)
    
    for generation in range(generations):
        new_population = []
        
        # 保留精英个体
        elite=min(population,key=calculate_fitness)
        new_population.append(elite)
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)
            
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])
        
        population = new_population
    
    best_individual = min(population, key=lambda x: calculate_fitness(x))
    best_x = int(''.join(map(str, best_individual)), 2)
    
    return best_x

# 设置参数并运行遗传算法
population_size = 50
chromosome_length = 8
generations = 100
tournament_size = 3
crossover_rate = 0.5
mutation_rate = 0.5
best_solution = genetic_algorithm(population_size, chromosome_length, generations, tournament_size, crossover_rate, mutation_rate)

print("最优解为:", best_solution)
