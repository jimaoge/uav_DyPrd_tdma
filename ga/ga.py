import math
import pickle
import random
import time
import numpy as np
from matplotlib import pyplot as plt
import os
import json
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入配置
from config.config import config

# 定义一些参数
N_2 = config.GA_PARAMS['N^2']  # 你的 N_2 值
POP_SIZE = config.GA_PARAMS['population_size']
PC = config.GA_PARAMS['probability_of_cross']  # 交叉概率
PM = config.GA_PARAMS['probability_of_mutate']  # 变异概率
N_GENERATIONS = config.GA_PARAMS['number_of_generation']  # 主算法循环次数
NODE_NUM = config.GA_PARAMS['number_of_node']
SLOT_NUM = config.GA_PARAMS['number_of_slot']


def load_topology_from_json(filename):
    """
    从JSON文件加载拓扑数据
    Args:
        filename: JSON文件名
    Returns:
        neib_list: 二跳邻居矩阵列表
        weight_list: 权重列表
        topology_info: 拓扑信息字典
    """
    import json
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # 验证数据格式
    if 'topologies' not in data:
        raise ValueError("JSON文件中必须包含'topologies'键")
    
    # 获取节点数和时隙数（如果提供）
    node_count = data.get('node_count', NODE_NUM)
    slot_count = data.get('slot_count', SLOT_NUM)
    
    # 读取拓扑信息
    topologies = data['topologies']
    neib_list = []
    weight_list = []
    
    print(f"从文件 {filename} 加载了 {len(topologies)} 个拓扑结构")
    print(f"节点数: {node_count}, 时隙数: {slot_count}")
    
    for i, topo in enumerate(topologies):
        matrix = topo['one_hop_matrix']
        weight = topo.get('weight', 1.0/len(topologies))  # 默认等权重
        
        # 验证矩阵维度
        if len(matrix) != node_count or any(len(row) != node_count for row in matrix):
            raise ValueError(f"拓扑 {i} 的邻接矩阵维度不正确")
        
        # 计算二跳邻居矩阵
        two_hop_matrix = one_two_neighbors(matrix)
        
        neib_list.append(two_hop_matrix)
        weight_list.append(weight)
        
        print(f"  拓扑 {i} ({topo.get('name', f'topology_{i}')}): 权重={weight}")
    
    # 归一化权重
    total_weight = sum(weight_list)
    if total_weight > 0:
        weight_list = [w/total_weight for w in weight_list]
    
    print(f"权重归一化后: {weight_list}")
    
    return neib_list, weight_list, data

# 根据一跳邻居矩阵生成两条邻居矩阵
def one_two_neighbors(one_hop_neighbors):
    num_nodes = len(one_hop_neighbors)
    two_hop_neighbors = [[0] * num_nodes for _ in range(num_nodes)]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if one_hop_neighbors[i][j] == 1:
                two_hop_neighbors[i][j] = 1
                for k in range(num_nodes):
                    if one_hop_neighbors[j][k] == 1 and i != k:
                        two_hop_neighbors[i][k] = 1

    return two_hop_neighbors

# 表示遗传算法中的一个个体（时隙分配方案），每个个体有一个二进制序列（长度81=9节点×9时隙），包含适应度、选择概率等属性
class Individual:
    def __init__(self, m_Sequence):
        self.Sequence = m_Sequence
        self.Fitness = 0
        self.P_Fitness = 0
        self.Sum_P_Fitness = 0

    def Get_Sequence(self):
        return self.Sequence

    def chaFitness(self, m_Fitness):
        self.Fitness = m_Fitness

    def Get_Fitness(self):
        return self.Fitness

    def Get_P_Fitness(self):
        return self.P_Fitness

    def Get_Sum_P_Fitness(self):
        return self.Sum_P_Fitness

    def set_P_Fitness(self, mP_Fitness):
        self.P_Fitness = mP_Fitness

    def set_Sum_P_Fitness(self, mSum_P_Fitness):
        self.Sum_P_Fitness = mSum_P_Fitness

nowPopulation = []
midPopulation = []
nextPopulation = []
# fit = []

# 随机生成初始种群，每个个体是长度为81的随机二进制序列
def initialize_random():
    for i in range(POP_SIZE):
        X = [0] * N_2
        for j in range(N_2):
            if custom_random() <= 0.5:
                X[j] = 0
            else:
                X[j] = 1
        indivi = Individual(X)
        nowPopulation.append(indivi)
    print(f"随机生成初始种群");

# 自定义随机数生成函数，范围(0, 1)
def custom_random():
    N = random.randint(0, 999)
    return N / 1000.0

# 计算个体的适应度，考虑多个拓扑结构的加权适应度，
# 包含三个部分：
# 激活时隙奖励（+1分）
# 邻居节点冲突惩罚（-2×81分）
# 全零节点惩罚（-81分）
def cal_fitness(neib_list, weight_list=None):
    # 检查参数长度一致性
    if weight_list is not None and len(weight_list) != len(neib_list):
        raise ValueError(f"weight_list长度({len(weight_list)})与neib_list长度({len(neib_list)})不一致")
    
    for i in range(POP_SIZE):
        seq = nowPopulation[i].Get_Sequence()
        weighted_fitness = 0.0
        
        # 如果没有提供weight_list，则创建等权重列表
        if weight_list is None:
            weight_list = [1.0 / len(neib_list)] * len(neib_list)
        
        # 遍历三个拓扑和对应的权重
        for neib_mat, weight in zip(neib_list, weight_list):
            topo_fitness = 0.0
            
            # 1. 激活时隙奖励
            active_count = sum(seq)
            topo_fitness += active_count
            
            # 2. 邻居节点冲突惩罚
            for slot in range(SLOT_NUM):
                slot_start = slot * NODE_NUM
                slot_end = (slot + 1) * NODE_NUM
                slot_vals = seq[slot_start:slot_end]
                
                # 检查二跳内邻居冲突
                for ii in range(NODE_NUM):
                    for jj in range(ii + 1, NODE_NUM):
                        if neib_mat[ii][jj] == 1:  # 使用当前拓扑的邻接矩阵
                            if slot_vals[ii] == 1 and slot_vals[jj] == 1:
                                topo_fitness -= 2 * N_2  # 冲突惩罚
            
            # 3. 全零节点惩罚
            zero_nodes = 0
            for node in range(NODE_NUM):
                all_zeros = True
                for slot in range(SLOT_NUM):
                    idx = slot * NODE_NUM + node
                    if seq[idx] != 0:
                        all_zeros = False
                        break
                if all_zeros:
                    zero_nodes += 1
                    topo_fitness -= N_2  # 全零节点惩罚
            
            # 加权当前拓扑的适应度
            weighted_fitness += weight * topo_fitness
        
        nowPopulation[i].chaFitness(weighted_fitness)

# 根据个体适应度决定选择概率
def cal_P_fitness():
    sum_fitness = 0  # 适应度累计值
    temp = 0  # 临时存放适应度概率
    for i in range(POP_SIZE):
        sum_fitness += nowPopulation[i].Get_Fitness()

    for j in range(POP_SIZE):
        temp = nowPopulation[j].Get_Fitness() / sum_fitness
        nowPopulation[j].set_P_Fitness(temp)

# TODO: 实现适应度累计概率计算的代码
def cal_Sum_fitness():
    summation = 0  # 累加值存放
    for i in range(POP_SIZE):
        summation += nowPopulation[i].Get_P_Fitness()
        nowPopulation[i].set_Sum_P_Fitness(summation)

# 选择操作，保留25%的最佳个体，用轮盘赌选择剩下的75%
def select():
    max_fitness = nowPopulation[0].Get_Fitness()
    max_id = 0
    for p in range(POP_SIZE):
        if max_fitness < nowPopulation[p].Get_Fitness():
            max_fitness = nowPopulation[p].Get_Fitness()
            max_id = p

    for i in range(POP_SIZE // 4):
        midPopulation.append(nowPopulation[max_id])

    newPO_SIZE = POP_SIZE - POP_SIZE // 4
    array = [random.uniform(0.0, 1.0) for _ in range(newPO_SIZE)]

    midFitness = nowPopulation[0].Get_Sum_P_Fitness()
    # 轮盘进行选择
    for j in range(newPO_SIZE):
        if array[j] < midFitness:
            midPopulation.append(nowPopulation[0])  # 加入到中间种群
        else:
            for i in range(1, POP_SIZE):
                if array[j] >= nowPopulation[i - 1].Get_Sum_P_Fitness() and array[j] <= nowPopulation[
                    i].Get_Sum_P_Fitness():
                    midPopulation.append(nowPopulation[i])  # 加入到中间种群
                    break

    nowPopulation.clear()  # 清空nowpopulation

# 在时隙边界（9的倍数位置）进行单点交叉
def crossover():
    num = 0  # 记录次数
    while num < POP_SIZE - 1:
        ranC = random.random()
        array1 = midPopulation[num].Get_Sequence()[:]
        array2 = midPopulation[num + 1].Get_Sequence()[:]

        if ranC <= PC:  # 如果随机小数小于交叉概率，则进行交叉操作
            # crosspos = random.randint(1, N_2 - 1)  # 随机确定交叉点1-9
            crosspos = 9 * random.randint(1, 8)  # 随机确定交叉点1-9

            # 提取交叉序列片段
            new_arr1 = array1[:crosspos] + array2[crosspos:]
            new_arr2 = array2[:crosspos] + array1[crosspos:]

            newChild1 = Individual(new_arr1)
            newChild2 = Individual(new_arr2)
            nextPopulation.extend([newChild1, newChild2])
        else:
            nextPopulation.extend([midPopulation[num], midPopulation[num + 1]])

        num += 2

    midPopulation.clear()  # 清空 midPopulation

# def crossover():
#     num = 0
#     while num < POP_SIZE - 1:
#         # 获取父代序列
#         seq1 = midPopulation[num].Get_Sequence()[:]
#         seq2 = midPopulation[num + 1].Get_Sequence()[:]
        
#         # 创建子代序列
#         child1_seq = []
#         child2_seq = []
        
#         # 以时隙为单位进行交叉
#         for slot in range(SLOT_NUM):
#             slot_start = slot * NODE_NUM
#             slot_end = (slot + 1) * NODE_NUM
            
#             # 使用相同的随机数决定是否交换该时隙
#             if random.random() < PC:
#                 # 交换整个时隙的分配
#                 child1_seq.extend(seq2[slot_start:slot_end])
#                 child2_seq.extend(seq1[slot_start:slot_end])
#             else:
#                 child1_seq.extend(seq1[slot_start:slot_end])
#                 child2_seq.extend(seq2[slot_start:slot_end])
        
#         # 创建子代个体
#         newChild1 = Individual(child1_seq)
#         newChild2 = Individual(child2_seq)
#         nextPopulation.extend([newChild1, newChild2])
        
#         num += 2
    
#     midPopulation.clear()


# 随机交换两个位置的值
def mutation():
    for i in range(POP_SIZE):
        temp_seq = nextPopulation[i].Get_Sequence()[:]
        ranM = random.random()

        if ranM <= PM:
            # 随机选择两个位置进行交换
            mu_arr = random.sample(range(N_2), 2)
            temp = temp_seq[mu_arr[0]]
            temp_seq[mu_arr[0]] = temp_seq[mu_arr[1]]
            temp_seq[mu_arr[1]] = temp

            muChild = Individual(temp_seq)
            nowPopulation.append(muChild)
        else:
            nowPopulation.append(nextPopulation[i])

    nextPopulation.clear()

def select_top_individuals(P, top_n=10):
    # 根据适应度选择前top_n个最好的个体
    sorted_P = sorted(P, key=lambda x: x.Fitness, reverse=True)
    return sorted_P[:top_n]

# 用优秀个体+随机个体初始化种群，目的是加速收敛
def initialize_with_top_individuals(top_individuals=[]):
    nowPopulation.extend(top_individuals)
    remaining_size = POP_SIZE - len(top_individuals)
    for i in range(remaining_size):
        X = [random.randint(0, 1) for _ in range(N_2)]
        indivi = Individual(X)
        nowPopulation.append(indivi)

# 主遗传算法循环，执行N_GENERATIONS代进化，记录每代最佳和平均适应度，保存结果和收敛曲线
def genetic_algorithm(neib_list, weight_list, k = 20, P = None, run_id = 0):
    global nowPopulation  # 使用global声明，以便修改全局变量
    # 判断是否提供了种群P，若没有则初始化种群
    nowPopulation = []  # 确保种群从空开始
    if P is not None:
        # 从提供的种群P中选择k个适应度最高的个体
        top_individuals = select_top_individuals(P, k)
        # 使用带有顶尖个体的初始化函数
        initialize_with_top_individuals(top_individuals)
    else:
        # 没有提供种群P时，完全随机初始化种群
        initialize_random()
        save_population(nowPopulation)
        
    maxFitTemp = 0 - N_2
    maxFitIndex = 0
    T = N_GENERATIONS
    # 添加记录适应度历史的列表
    max_fitness_history = []  # 记录每次迭代最优个体的适应度
    avg_fitness_history = []  # 记录每次迭代种群的平均适应度

    # 在while循环开始之前初始化全局最大适应度
    global_max_fit = float('-inf')
    global_max_fit_index = -1

    while T:
        T -= 1
        # 定期打印训练信息
        if T % 100 == 0:
            print(f"遗传算法训练迭代剩余{T}次, 目前最大适应度：{global_max_fit}")        
        # 1. 计算当前种群的适应度
        if weight_list is not None and len(weight_list) > 0:  # weight_list不为None且非空
            cal_fitness(neib_list, weight_list)
        else:  # weight_list为None或空列表
            cal_fitness(neib_list)
        
        # 2. 更新全局最大适应度
        maxFitTemp = float('-inf')
        for s in range(POP_SIZE):
            if maxFitTemp < nowPopulation[s].Get_Fitness():
                maxFitTemp = nowPopulation[s].Get_Fitness()
                maxFitIndex = s
        
        if maxFitTemp > global_max_fit:
            global_max_fit = maxFitTemp
        
        # 3. 计算并记录当前种群的平均适应度
        avg_fitness = sum(indiv.Get_Fitness() for indiv in nowPopulation) / POP_SIZE
        avg_fitness_history.append(avg_fitness)
        max_fitness_history.append(global_max_fit)
        
        # 4. 执行遗传操作（此时记录的是操作前的适应度）
        cal_P_fitness()
        cal_Sum_fitness()
        select()
        crossover()
        mutation()



    # 输出算法运行的最终结果和运行时间
    sequence_str = ', '.join(map(str, nowPopulation[maxFitIndex].Get_Sequence()))
    # print(f"最佳解：{sequence_str}  最大适应度：{maxFit}")


    # 转换为矩阵
    assert len(nowPopulation[maxFitIndex].Get_Sequence()) == N_2, "列表长度不正确"
    sequence = nowPopulation[maxFitIndex].Get_Sequence()
    matrix = np.array(sequence).reshape(SLOT_NUM, NODE_NUM)
    print(f"遗传算法求解的时隙分配矩阵：\n{matrix}")
    
    result_path = config.DATA_PATHS['saved_slots_matrix_path']
    # 确保结果目录存在
    os.makedirs('result', exist_ok=True)
    
    # 创建带时间戳的唯一文件名
    timestamp = 1
    
    # 保存适应度历史数据到JSON文件
    fitness_data = {
        'max_fitness_history': max_fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'run_id': run_id,
        'timestamp': timestamp,
        'parameters': {
            'POP_SIZE': POP_SIZE,
            'PC': PC,
            'PM': PM,
            'N_GENERATIONS': N_GENERATIONS,
            'NODE_NUM': NODE_NUM,
            'SLOT_NUM': SLOT_NUM
        }
    }
    
    data_filename = os.path.join(result_path, f"fitness_data_{timestamp}_run{run_id}.json")
    with open(data_filename, 'w') as f:
        json.dump(fitness_data, f, indent=2)
    
    print(f"适应度历史数据已保存至: {data_filename}")
    
    # 然后绘制并保存收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(max_fitness_history, label='Max Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Convergence (Run {run_id})')
    plt.legend()
    plt.grid(True)
    
    plot_filename = os.path.join(result_path, f"fitness_convergence_{timestamp}_run{run_id}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"收敛曲线已保存至: {plot_filename}")
    plt.show()
    
    return matrix, nowPopulation, max_fitness_history[-1]

def save_population(population, filename = config.DATA_PATHS['saved_population_file_name']):
    # 获取保存路径
    save_path = config.DATA_PATHS['saved_population_path']
    
    # 确保目录存在
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_filename = os.path.join(save_path, filename)
    else:
        # 如果没有配置路径，使用当前目录
        full_filename = filename
    
    with open(full_filename, 'wb') as f:
        pickle.dump(population, f)
    print(f"种群已保存至: {full_filename}")
    
# 根据节点的三维坐标生成邻接矩阵
# x,y,z三个列表的长度相同，表示网络中的节点数量
def topy(x, y, z):
    num_nodes = len(x)
    link_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distance = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)
                if distance < config.ENV_PARAMS['communication_range']:
                    link_matrix[i][j] = 1
                    link_matrix[j][i] = 1

    # print("Link Matrix:")
    # print(link_matrix)

    return link_matrix

# 从文件加载种群
def load_population(filename = config.DATA_PATHS['saved_population_file_name']):
    # 从配置中获取路径
    load_path = config.DATA_PATHS['saved_population_path']
    full_path = os.path.join(load_path, filename)
    
    # 确保目录存在（如果是保存操作）
    # 对于加载操作，这里我们只检查文件是否存在
    if not os.path.exists(full_path):
        print(f"错误: 文件 {full_path} 不存在")
        raise FileNotFoundError(f"文件 {full_path} 不存在")
    
    with open(full_path, 'rb') as f:
        population = pickle.load(f)
        print(f"种群已从 {full_path} 加载")
        return population




def save_result_to_json(matrix, topology_info=None, filename="result_allocation.json"):
    """
    保存时隙分配结果到JSON文件
    Args:
        matrix: 时隙分配矩阵 (9x9)
        topology_info: 拓扑信息
        filename: 输出文件名
    """
    import json
    from datetime import datetime
    
    result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "allocation_matrix": matrix.tolist() if hasattr(matrix, 'tolist') else matrix,
        "parameters": {
            "NODE_NUM": NODE_NUM,
            "SLOT_NUM": SLOT_NUM,
            "POP_SIZE": POP_SIZE,
            "PC": PC,
            "PM": PM,
            "N_GENERATIONS": N_GENERATIONS
        }
    }
    
    if topology_info:
        result["topology_info"] = topology_info
    
    with open(f"result/{filename}", 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"时隙分配结果已保存到 result/{filename}")

if __name__ == "__main__":
    print("START")

    # neib = [
    #         [0, 1, 1, 0, 1, 0, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #         [1, 0, 0, 0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0]
    #     ]

    # neib1 = [
    #         [0, 1, 1, 0, 1, 1, 0, 0, 0], #
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #         [1, 0, 0, 0, 0, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0], #
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0]
    #     ]

    # neib2 = [
    #         [0, 1, 1, 0, 1, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0], #
    #         [1, 0, 0, 1, 0, 1, 0, 0, 0], #
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0]
    #     ]

    # neib3 = [
    #         [0, 1, 1, 0, 1, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [1, 0, 0, 1, 0, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 1, 0, 0, 0, 0, 0, 1],  # 变化：添加了(7,8)边
    #         [0, 0, 0, 0, 0, 0, 1, 1, 0]   # 变化：添加了(8,7)边
    #     ]    

    # neib4 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 0],  # 变化：删除了(0,1)边
    #         [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 变化：删除了(1,0)边
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [1, 0, 0, 1, 0, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 1, 0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 1, 1, 0]
    #     ]

    # neib5 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 变化：添加了(1,4)边
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [1, 1, 0, 1, 0, 1, 0, 0, 0],  # 变化：添加了(4,1)边
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 1, 0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 1, 1, 0]
    #     ]

    # neib6 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 变化：删除了(2,7)边
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [1, 1, 0, 1, 0, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 0, 0, 0, 0, 0, 0, 1],  # 变化：删除了(7,2)边
    #         [0, 0, 0, 0, 0, 0, 1, 1, 0]
    #     ]
    # neib7 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 1],  # 变化：添加了(0,8)边
    #         [0, 0, 0, 0, 1, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [1, 1, 0, 1, 0, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 1, 1, 0]   # 变化：添加了(8,0)边
    #     ]
    # neib8 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 1],
    #         [0, 0, 0, 0, 1, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 变化：删除了(3,4)边
    #         [1, 1, 0, 0, 0, 1, 0, 0, 0],  # 变化：删除了(4,3)边
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 1, 1, 0]
    #     ]
    # neib9 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 1],
    #         [0, 0, 0, 0, 1, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0],
    #         [1, 1, 0, 0, 0, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 1, 0, 1, 1, 0],  # 变化：添加了(5,7)边
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 0, 0, 0, 1, 0, 0, 1],  # 变化：添加了(7,5)边
    #         [1, 0, 0, 0, 0, 0, 1, 1, 0]
    #     ]
    
    # # 获得两跳邻居邻接图
    # neib_2 = one_two_neighbors(neib)
    # neib_21 = one_two_neighbors(neib1)
    # neib_22 = one_two_neighbors(neib2)
    # neib_23 = one_two_neighbors(neib3)
    # neib_24 = one_two_neighbors(neib4)
    # neib_25 = one_two_neighbors(neib5)
    # neib_26 = one_two_neighbors(neib6)
    # neib_27 = one_two_neighbors(neib7)
    # neib_28 = one_two_neighbors(neib8)
    # neib_29 = one_two_neighbors(neib9)

    # neib_list = [neib_2, neib_21, neib_22, neib_23, neib_24, neib_25, neib_26, neib_27, neib_28, neib_29]
    # weight_list = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    # neib_list = [neib_2, neib_21, neib_22]
    # 假设应用周期是10s，三种拓扑分别占1s,2s,7s

    # 从JSON文件加载拓扑数据
    try:
        # 可以设置默认文件名，或者从命令行参数获取
        import sys
        if len(sys.argv) > 1:
            filename = sys.argv[1]
        else:
            filename = "neighbor_matrices/neighbor_matrices.json"
        
        neib_list, weight_list, topology_info = load_topology_from_json(filename)
        
        # 输出拓扑信息
        print(f"成功加载 {len(neib_list)} 个拓扑结构")
        print(f"网络节点数: {NODE_NUM}, 时隙数: {SLOT_NUM}")
        
    except FileNotFoundError:
        print(f"错误: 文件 {filename} 未找到")
        print("将使用硬编码的拓扑结构...")
        # 回退到硬编码的拓扑（可选）
        neib1 = [[0, 1, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 1],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0]]
        
        neib2 = [[0, 1, 1, 0, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 0, 1, 0, 0],
                [1, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0, 1],
                [0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0]]
        
        neib_list = [one_two_neighbors(neib1), one_two_neighbors(neib2)]
        weight_list = [0.5, 0.5]
        
    #每次保留适应度最高的u个个体
    u = 20
    # 随机初始化种群，保存迭代后的种群
    m, P, TEMP = genetic_algorithm(neib_list, weight_list, u)
    save_population(P)
    save_result_to_json(m, topology_info)
    # 加载固定的种群
    # P = load_population()
    # genetic_algorithm(neib_list, weight_list,u, P)
    
    # 加载固定的种群并保存迭代后的种群
    # P = load_population()
    # m, newP , TEMP= genetic_algorithm(neib_list, weight_list, u, P)
    # save_population(newP)
    
    # 随机初始化种群，保存该初始化种群
    # genetic_algorithm(neib_list, weight_list,u)
    
    # test()





