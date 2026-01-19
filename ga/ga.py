import pickle
import random
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from pathlib import Path

nowPopulation = []  # 当前种群
midPopulation = []
nextPopulation = []


# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入配置
from config.config import config

# 定义一些参数
N_2 = config.GA_PARAMS['N^2']  # NODE_NUM * SLOT_NUM, 个体的二进制序列长度
POP_SIZE = config.GA_PARAMS['population_size']
PC = config.GA_PARAMS['probability_of_cross']  # 交叉概率
PM = config.GA_PARAMS['probability_of_mutate']  # 变异概率
N_GENERATIONS = config.GA_PARAMS['number_of_generation']  # 主算法循环次数
NODE_NUM = config.GA_PARAMS['number_of_node']
SLOT_NUM = config.GA_PARAMS['number_of_slot']
assert N_2 == NODE_NUM * SLOT_NUM, f"参数错误:N_2={N_2} 必须等于 NODE_NUM({NODE_NUM}) * SLOT_NUM({SLOT_NUM})"


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

# 根据一跳邻居矩阵生成二跳邻居矩阵（一跳+二跳邻居，对角线置0，无自冲突）
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
        two_hop_neighbors[i][i] = 0  # 对角线置0，节点自己不是自己的邻居
    return two_hop_neighbors

# 表示遗传算法中的一个个体（时隙分配方案），每个个体有一个二进制序列（长度81=9节点*9时隙），包含适应度、选择概率等属性
class Individual:
    def __init__(self, m_Sequence):
        self.Sequence = m_Sequence
        self.Fitness = 0
        self.P_Fitness = 0
        self.Sum_P_Fitness = 0

    def Get_Sequence(self):
        return self.Sequence

    def set_Fitness(self, m_Fitness):
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

# 随机生成初始种群，每个个体是长度为81的随机二进制序列
def initialize_random():
    global nowPopulation
    nowPopulation.clear()
    for i in range(POP_SIZE):
        X = [random.randint(0, 1) for _ in range(N_2)]
        indivi = Individual(X)
        nowPopulation.append(indivi)
    print(f"随机生成初始种群");

# 考虑拓扑推演结果的多个拓扑, 计算个体的适应度
# 首先计算在每个拓扑上的:
# 初始值为激活时隙奖励 - 82，因为激活时隙奖励最大为81，保证为负数
# 激活时隙奖励, 有一个1就+1分, 不考虑冲突与否
# 邻居节点冲突惩罚，遍历所有时隙，发现一个冲突就-2分
# 全零节点惩罚，遍历所有节点，发现一个全零就-3分
# 对所有拓扑的适应度求和后求平均得到个体在该拓扑推演结果上的适应度
# 该函数目前只存在cal_fitness(neib_list)的调用，即weight_list均为None, 相当于等权重做平均
def cal_fitness(neib_list, weight_list=None):
    global nowPopulation
    # 参数校验
    if not neib_list:
        raise ValueError("邻接矩阵列表neib_list不能为空，至少传入1个拓扑邻接矩阵")
    if weight_list is not None and len(weight_list) != len(neib_list):
        raise ValueError(f"weight_list长度({len(weight_list)})与neib_list长度({len(neib_list)})不一致")
    # 校验邻接矩阵维度合法性:必须是 NODE_NUM * NODE_NUM 的二维矩阵
    for idx, mat in enumerate(neib_list):
        if len(mat) != NODE_NUM or any(len(row) != NODE_NUM for row in mat):
            raise ValueError(f"第{idx+1}个邻接矩阵维度错误，要求是{NODE_NUM}*{NODE_NUM}的二维矩阵")

    if weight_list is None:
        weight_list = [1.0 / len(neib_list)] * len(neib_list)
    # NODE_NUM = SLOT_NUM = 9, N_2 = NODE_NUM * SLOT_NUM
    PUNISH_CONFLICT = 2  # 冲突惩罚值
    PUNISH_ZERO_NODE = 3     # 全零节点惩罚值

    for i in range(POP_SIZE):
        seq = nowPopulation[i].Get_Sequence()
        weighted_fitness = 0.0

        # 激活时隙奖励，每个个体独立计算
        active_count = sum(seq)
        
        # 遍历所有拓扑+权重
        for neib_mat, weight in zip(neib_list, weight_list):
            # 这里减去(N_2 + 1)是保证所有适应度均为负数,因为后续函数存在“分母为适应度求和”的情况,避免除零错误
            topo_fitness = active_count - N_2 - 1

            # 邻居节点冲突惩罚
            for slot in range(SLOT_NUM):
                slot_base_idx = slot * NODE_NUM
                for ii in range(NODE_NUM):
                    val_ii = seq[slot_base_idx + ii]
                    if val_ii != 1:
                        continue
                    for jj in range(ii + 1, NODE_NUM):
                        if neib_mat[ii][jj] == 1 and seq[slot_base_idx + jj] == 1:
                            topo_fitness -= PUNISH_CONFLICT

            # 全零节点惩罚
            for node in range(NODE_NUM):
                all_zeros = True
                for slot in range(SLOT_NUM):
                    if seq[slot * NODE_NUM + node] != 0:
                        all_zeros = False
                        break
                if all_zeros:
                    # 全零惩罚
                    topo_fitness -= PUNISH_ZERO_NODE

            # 加权累加当前拓扑的适应度
            weighted_fitness += weight * topo_fitness
        nowPopulation[i].set_Fitness(weighted_fitness)

# 根据个体适应度决定选择概率
def cal_P_fitness():
    global nowPopulation
    # 找到最小适应度（最负的值）
    min_fitness = min(nowPopulation[i].Get_Fitness() for i in range(POP_SIZE))
    
    # 将所有适应度平移为正数
    offset = -min_fitness + 1  # 确保最小值为1
    adjusted_fitness_list = []
    
    for i in range(POP_SIZE):
        adjusted_fitness = nowPopulation[i].Get_Fitness() + offset
        adjusted_fitness_list.append(adjusted_fitness)
    
    # 3. 计算总和
    sum_fitness = sum(adjusted_fitness_list)
    
    # 4. 计算概率
    for i in range(POP_SIZE):
        p = adjusted_fitness_list[i] / sum_fitness
        nowPopulation[i].set_P_Fitness(p)

# TODO: 实现适应度累计概率计算的代码
def cal_Sum_fitness():
    global nowPopulation
    summation = 0  # 累加值存放
    for i in range(POP_SIZE):
        summation += nowPopulation[i].Get_P_Fitness()
        nowPopulation[i].set_Sum_P_Fitness(summation)

# 选择操作，从nowPopulation保留25%的最佳个体，轮盘D选择剩下的75%，最后构成midPopulation
def select():
    global nowPopulation, midPopulation
    #清空中间种群，防止上一轮数据残留
    midPopulation.clear()
    # 对种群按适应度从高到低排序，得到排序后的索引列表
    sorted_pop_ids = sorted(range(POP_SIZE), key=lambda x: nowPopulation[x].Get_Fitness(), reverse=True)
    # 取前 25% 的最优个体
    elite_num = POP_SIZE // 4
    for i in range(elite_num):
        elite_id = sorted_pop_ids[i]
        midPopulation.append(nowPopulation[elite_id])

    # 计算需要选择的普通个体数量
    newPO_SIZE = POP_SIZE - elite_num
    array = [random.uniform(0.0, 1.0) for _ in range(newPO_SIZE)]

    for j in range(newPO_SIZE):
        rand_val = array[j]
        # 遍历种群匹配累计概率
        select_flag = False
        for i in range(POP_SIZE):
            if rand_val <= nowPopulation[i].Get_Sum_P_Fitness():
                midPopulation.append(nowPopulation[i])
                select_flag = True
                break
        # 兜底逻辑:防止随机数=1.0时匹配不到（边界防护）
        if not select_flag:
            midPopulation.append(nowPopulation[-1])
            
    nowPopulation.clear()  # 清空原种群

# 对于select生成的midPopulation, 在时隙边界（9的倍数位置）进行单点交叉，最终生成nextPopulation
def crossover():
    global midPopulation, nextPopulation
    num = 0  # 记录次数
    nextPopulation.clear() # 前置清空，防止残留
    while num < POP_SIZE:
        ranC = random.random()
        # 处理成对个体
        if num < POP_SIZE - 1:
            array1 = midPopulation[num].Get_Sequence()[:]
            array2 = midPopulation[num + 1].Get_Sequence()[:]

            if ranC <= PC:
                crosspos = 9 * random.randint(1, 8)  # 时隙边界交叉
                new_arr1 = array1[:crosspos] + array2[crosspos:]
                new_arr2 = array2[:crosspos] + array1[crosspos:]
                newChild1 = Individual(new_arr1)
                newChild2 = Individual(new_arr2)
                nextPopulation.extend([newChild1, newChild2])
            else:
                nextPopulation.extend([midPopulation[num], midPopulation[num + 1]])
            num += 2
        else:
            # 处理最后一个落单的个体，直接保留不交叉
            nextPopulation.append(midPopulation[num])
            num += 1
    midPopulation.clear()  # 清空 midPopulation

# 对于crossover生成的nextPopulation, 随机交换两个位置的值，最终赋值给nowPopulation
def mutation():
    global nowPopulation, nextPopulation
    nowPopulation.clear() # 前置清空，防止残留
    for i in range(POP_SIZE):
        temp_seq = nextPopulation[i].Sequence[:]
        ranM = random.random()

        if ranM <= PM:
            # 随机选择1个位置进行二进制翻转
            mu_pos = random.randint(0, N_2 -1)
            temp_seq[mu_pos] = 1 - temp_seq[mu_pos] # 0→1，1→0

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
    global nowPopulation
    nowPopulation.clear()
    nowPopulation.extend(top_individuals)
    remaining_size = POP_SIZE - len(top_individuals)
    for i in range(remaining_size):
        X = [random.randint(0, 1) for _ in range(N_2)]
        indivi = Individual(X)
        nowPopulation.append(indivi)

# 主遗传算法循环，执行N_GENERATIONS代进化，记录每代最佳和平均适应度，保存结果和收敛曲线
def genetic_algorithm(neib_list, weight_list, k = 20, P = None, run_id = 0):
    global nowPopulation, midPopulation, nextPopulation
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
        # save_population(nowPopulation)
    
    maxFitIndex = 0
    T = N_GENERATIONS
    # 添加记录适应度历史的列表
    max_fitness_history = []  # 记录每次迭代最优个体的适应度
    avg_fitness_history = []  # 记录每次迭代种群的平均适应度

    # 在while循环开始之前初始化全局最大适应度
    global_max_fit = float('-inf')
    global_max_fit_index = -1

    while T: 
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
            global_max_fit_index = maxFitIndex  # 新增这一行，记录全局最优个体的索引
        
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
        # 定期打印训练信息
        if T % 10 == 0:
            print(f"遗传算法训练迭代剩余{T}次, 目前最大适应度:{global_max_fit}, 恢复偏移后为:{global_max_fit + 82}")  
        T -= 1     


    # 输出算法运行的最终结果和运行时间
    # sequence_str = ', '.join(map(str, nowPopulation[global_max_fit_index].Get_Sequence()))
    # print(f"最佳解:{sequence_str},全局最大适应度:{global_max_fit}")


    # 转换为矩阵
    assert len(nowPopulation[maxFitIndex].Get_Sequence()) == N_2, "列表长度不正确"
    sequence = nowPopulation[global_max_fit_index].Get_Sequence() # 返回全局最优索引
    matrix = np.array(sequence).reshape(SLOT_NUM, NODE_NUM)

    
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
    #         [0, 1, 1, 0, 0, 0, 0, 0, 1],  # 变化:添加了(7,8)边
    #         [0, 0, 0, 0, 0, 0, 1, 1, 0]   # 变化:添加了(8,7)边
    #     ]    

    # neib4 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 0],  # 变化:删除了(0,1)边
    #         [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 变化:删除了(1,0)边
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
    #         [0, 0, 0, 0, 1, 0, 0, 1, 0],  # 变化:添加了(1,4)边
    #         [1, 0, 0, 0, 0, 0, 0, 1, 0],
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [1, 1, 0, 1, 0, 1, 0, 0, 0],  # 变化:添加了(4,1)边
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 1, 0, 0, 0, 0, 0, 1],
    #         [0, 0, 0, 0, 0, 0, 1, 1, 0]
    #     ]

    # neib6 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 变化:删除了(2,7)边
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [1, 1, 0, 1, 0, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 0, 0, 0, 0, 0, 0, 1],  # 变化:删除了(7,2)边
    #         [0, 0, 0, 0, 0, 0, 1, 1, 0]
    #     ]
    # neib7 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 1],  # 变化:添加了(0,8)边
    #         [0, 0, 0, 0, 1, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [1, 1, 0, 1, 0, 1, 0, 0, 0],
    #         [1, 0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0, 0, 1, 1, 0]   # 变化:添加了(8,0)边
    #     ]
    # neib8 = [
    #         [0, 0, 1, 0, 1, 1, 0, 0, 1],
    #         [0, 0, 0, 0, 1, 0, 0, 1, 0],
    #         [1, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 变化:删除了(3,4)边
    #         [1, 1, 0, 0, 0, 1, 0, 0, 0],  # 变化:删除了(4,3)边
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
    #         [1, 0, 0, 0, 1, 0, 1, 1, 0],  # 变化:添加了(5,7)边
    #         [0, 0, 0, 1, 0, 1, 0, 0, 1],
    #         [0, 1, 0, 0, 0, 1, 0, 0, 1],  # 变化:添加了(7,5)边
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





