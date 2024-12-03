import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1])  # 粒子的位置
        self.velocity = np.random.uniform(-1, 1)                  # 粒子的速度
        self.best_position = self.position                         # 粒子历史最优位置
        self.best_value = float('inf')                             # 粒子历史最优值

    def evaluate(self, objective_function):
        value = objective_function(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = self.position

def objective_function(x):
    return x**2  # 优化目标函数

def pso(num_particles, bounds, num_iterations):
    particles = [Particle(bounds) for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')

    for _ in range(num_iterations):
        for particle in particles:
            particle.evaluate(objective_function)

            # 更新全局最优位置
            if particle.best_value < global_best_value:
                global_best_value = particle.best_value
                global_best_position = particle.best_position

        # 更新粒子的速度和位置
        for particle in particles:
            w = 0.5  # 惯性权重 一个较大的惯性权值有利于全局搜索，而一个较小的惯性权值则更利于局部搜索
            c1 = 2.0  # 自我认知参数 在算法搜索初期采用较大的c1值和较小的c2值,使粒子尽量发散到搜索空间即强调“个体独立意识”，而较少受到种群内其他粒子即“社会意识部分”的影响,以增加群内粒子的多样性。随着选代次数的增加,使c1线性递减, c2线性递增,从而加强了粒子向全局最优点的收敛能力。
            c2 = 2.0  # 社会认知参数 设置c1较大的值，会使粒子过多地在自身的局部范围内搜索，而较大的c2的值，则又会促使粒子过早收敛到局部最优值
            
            r1 = np.random.rand()
            r2 = np.random.rand()

            # 更新速度
            particle.velocity = (w * particle.velocity +
                                 c1 * r1 * (particle.best_position - particle.position) +
                                 c2 * r2 * (global_best_position - particle.position))
            
            # 更新位置
            particle.position += particle.velocity

            # 确保粒子在边界内
            particle.position = np.clip(particle.position, bounds[0], bounds[1])

    return global_best_position, global_best_value

# 参数设置
num_particles = 50  # 粒子数量
bounds = [-10, 10]   # 搜索边界
num_iterations = 100  # 迭代次数

best_position, best_value = pso(num_particles, bounds, num_iterations)
print(f'最佳位置: {best_position}, 最优值: {best_value}')
