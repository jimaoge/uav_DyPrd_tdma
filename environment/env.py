# 这个环境是计算TDMA时隙分配算法的方法是遗传算法

import glob
import os
import numpy as np
import pickle
import joblib
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import random
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入配置
from config.config import config

## 遗传算法
from lstm.lstm_model import TopologyPredictor
from ga.ga import genetic_algorithm, one_two_neighbors

# 设置计算设备（GPU或CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SelfOrganizingNetworkEnv:
    def __init__(self, data_path=config.DATA_PATHS['origin_traj_data']):
        # 初始化环境参数
        self.path = data_path  # 轨迹原始数据所在路径
        self.datafile_path = self.path / config.DATA_PATHS['traj_data_file_name']  # 拼接轨迹原始数据文件名
        self.data = pd.read_csv(self.datafile_path, comment='#')  # 轨迹原始数据
        print("原始轨迹数据已读取")
        self.scaler_path = config.DATA_PATHS['processed_traj_data']
        self.node_nums = config.ENV_PARAMS['uav_nums']
        self.slot_nums = config.ENV_PARAMS['slot_nums']
        self.cur_prd_start_time = 50  # 当前轮的开始时间tb
        self.next_prd_start_time = 50  # # 下一轮的开始时间td
        self.max_time = self.data['Time'].max()  # step里系统时间超过该时间会返回done=true,暂时没用
        self.pre_DyPrd = 2  # 上一轮的应用周期
        self.DyPrd = 2  # 当前轮的应用周期
        self.lstm_models = []
        self.T_sum = 0
        self.diff = []   # 本轮拓扑预测和实际拓扑邻接矩阵的差异数的数组，长度为DyPrd
        self.coll = 0   # 本轮时隙分配方案在真实环境下的时隙冲突总数量，由DyPrd个数相加而成
        self.slot_reward_ratio = config.DQN_PARAMS['slot_reward_ratio']
        self.DQN_k = config.DQN_PARAMS['DQN_k']
        self.scalers_list = []

    def load_scalers_list(self):
        """加载9个归一化器文件到列表中"""
        self.scalers_list = []
        for i in range(9):  # 节点ID 0-8
            scaler_path = Path(self.scaler_path) / f"scaler_node_{i}.pkl"
            scaler = joblib.load(scaler_path)  # 使用joblib加载
            self.scalers_list.append(scaler)

    def load_lstm_model(self):
        # 加载9个节点的LSTM预测模型
        for i in range(9):
            # 从config对象中获取LSTM模型参数
            input_size = config.LSTM_PARAMS['input_size']
            hidden_size = config.LSTM_PARAMS['hidden_size']
            num_layers = config.LSTM_PARAMS['num_layers']
            pred_len = config.LSTM_PARAMS['pred_len']
            dropout = config.LSTM_PARAMS['dropout']  # 现在直接从config获取
            
            # 创建模型实例，传递从config获取的参数
            model = TopologyPredictor(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                pred_len=pred_len,
                output_size=input_size,  # 输出维度与输入维度相同
                dropout=dropout,
                device=device
            ).to(device)
            
            # 构建模型文件路径
            lstm_path = config.DATA_PATHS['saved_lstm_models_path'] / f"model_node_{i}.pth"
            
            # 直接加载模型权重
            model.load_state_dict(torch.load(lstm_path, weights_only=True))
            # 如果需要在CPU上加载，使用下面这行注释掉的代码
            # model.load_state_dict(torch.load(lstm_path, map_location=torch.device('cpu')))
            
            # 将模型设置为评估模式
            model.eval()
            
            # 将模型添加到列表
            self.lstm_models.append(model)

    def reset_fix_time(self):
        # 重置环境状态
        self.cur_prd_start_time = 50
        self.next_prd_start_time = self.cur_prd_start_time
        # 重置应用周期DyPrd
        self.pre_DyPrd = 2
        self.DyPrd = 2
        # 重置记录的数据
        self.diff = []
        self.coll = 0
        self.T_sum = 0
        # 删除保存的时隙分配方案
        folder_path = config.DATA_PATHS['saved_slots_matrix_path']
        npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
        for file_path in npy_files:
            try:
                os.remove(file_path)
                print(f"文件 {file_path} 已被删除")
            except OSError as e:
                print(f"删除文件 {file_path} 时出错: {e}")
        print("所有 .npy 文件已被删除")
        return self.get_current_state()

    def reset_random_time(self):
        # 在可用轨迹范围内随机选择起始点
        max_start_time = config.DQN_PARAMS['max_time']
        self.cur_prd_start_time = random.randint(50, max_start_time - 1000)
        print(f"环境随机初始化开始时间为{self.cur_prd_start_time}s")
        self.next_prd_start_time = self.cur_prd_start_time
        # 重置应用周期DyPrd
        self.pre_DyPrd = 2
        self.DyPrd = 2
        # 重置记录的数据
        self.diff = []
        self.coll = 0
        self.T_sum = 0
        # 删除保存的时隙分配方案
        folder_path = config.DATA_PATHS['saved_slots_matrix_path']
        npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
        for file_path in npy_files:
            try:
                os.remove(file_path)
                print(f"文件 {file_path} 已被删除")
            except OSError as e:
                print(f"删除文件 {file_path} 时出错: {e}")
        print("所有 .npy 文件已被删除")
        return self.get_current_state()     
    
    # 执行一个动作并返回新的状态、奖励和完成标志
    def step(self, action):
        self.pre_DyPrd = self.DyPrd  # 保存上一轮的DyPrd
        
        if action == 0:
            self.DyPrd = 2
        if action == 1:
            self.DyPrd = 3
        if action == 2:
            self.DyPrd = 4
        if action == 3:
            self.DyPrd = 5
        if action == 4:
            self.DyPrd = 6
        if action == 5:
            self.DyPrd = 7
        if action == 6:
            self.DyPrd = 8
        if action == 7:
            self.DyPrd = 9
        if action == 8:
            self.DyPrd = 10
        # if action == 9:
        #     self.DyPrd = 11

        print(f"应用周期DyPrd：{self.DyPrd}")
        self.next_prd_start_time = min(self.cur_prd_start_time + self.DyPrd, self.max_time)

        # 预测拓扑并计算预测差异
        # predicted_topology为一个长度为DyPrd的列表，其中每个元素是该时间点的 (9, 9)的预测邻接矩阵
        predicted_topology = self.predict_topology(self.DyPrd)
        # rel_topology为一个长度为DyPrd的列表，其中每个元素是该时间点的 (9, 9)的实际邻接矩阵
        rel_topology = self.get_cur_link_matrix_list()
        
        # self.diff为包含所有时间点差异数的数组
        self.diff = self.get_diff(predicted_topology, rel_topology)
        # print(f"diff:{self.diff}")
        
        # 通过遗传算法生成时隙分配矩阵
        slot_allocation_matrix, population, max_fitness = genetic_algorithm(predicted_topology, weight_list = [])
        # 保存分配方案
        # self.save_slot(self.current_time + self.DyPrd, slot_allocation_matrix, config.DATA_PATHS['saved_slots_matrix_path'])

        # 计算奖励
        reward = self.calculate_reward(slot_allocation_matrix)
        # print(f"reward:{reward}")

        # 更新当前时间
        self.cur_prd_start_time = self.next_prd_start_time
        done = self.cur_prd_start_time >= self.max_time

        return self.get_current_state(), reward, done

    def calculate_link_dynamics(self, link_matrix_list):
        """
        计算链路动态度（消失、新增、保持的链路数）
        
        Args:
            link_matrix_list: 邻接矩阵列表, 长度为self.DQN_k
        
        Returns:
            链路动态度数组，形状为(self.DQN_k - 1, 3)，每行对应一个时帧变化
        """
        if len(link_matrix_list) != self.DQN_k:
            raise ValueError(f"Expected link_matrix_list length self.DQN_k, got {len(link_matrix_list)}")
        
        dynamics_array = []
        
        for i in range(1, len(link_matrix_list)):
            prev_matrix = link_matrix_list[i-1]
            curr_matrix = link_matrix_list[i]
            
            # 检查矩阵维度一致性
            if prev_matrix.shape != curr_matrix.shape:
                raise ValueError(f"Matrix dimensions mismatch at index {i}: "
                            f"{prev_matrix.shape} vs {curr_matrix.shape}")
            
            # 计算消失的链路数（上一时有，当前时无）
            disappeared = np.sum((prev_matrix == 1) & (curr_matrix == 0) & (np.eye(prev_matrix.shape[0]) == 0))
            
            # 计算新增的链路数（上一时无，当前时有）
            new_links = np.sum((prev_matrix == 0) & (curr_matrix == 1) & (np.eye(prev_matrix.shape[0]) == 0))
            
            # 计算保持的链路数（两时都有）
            maintained = np.sum((prev_matrix == 1) & (curr_matrix == 1) & (np.eye(prev_matrix.shape[0]) == 0))
            
            # 添加到结果数组
            dynamics_array.append([disappeared, new_links, maintained])
        
        return np.array(dynamics_array)


    def get_cur_link_matrix_list(self):
        """
        返回从self.cur_prd_start_time到self.next_prd_start_time-1
        一共self.DyPrd个时间点的真实邻接矩阵列表
        """
        return self.get_link_matrix_list(self.cur_prd_start_time, self.next_prd_start_time)

    def get_link_matrix_list(self, start_time, end_time):
        """
        返回从start_time到end_time-1时间段的真实邻接矩阵列表
        
        Args:
            start_time: 起始时间
            end_time: 结束时间（不包含）
        
        Returns:
            邻接矩阵列表，长度为(end_time - start_time)
        """
        duration = end_time - start_time
        
        if duration <= 0:
            raise ValueError(f"Invalid time range: start_time={start_time}, end_time={end_time}, duration={duration}")
        
        if start_time < 0 or end_time > self.max_time:
            raise ValueError(f"Time range [{start_time}, {end_time}) out of bounds [0, {self.max_time})")
        
        link_matrix_list = []
        
        for t in range(start_time, end_time):
            # 获取节点坐标数据
            node_positions = self.data[self.data['Time'] == t][['X', 'Y', 'Z']].values
            
            # 获取节点数量
            num_nodes = len(node_positions)
            
            # 检查节点数量是否与网络节点数一致
            if num_nodes != self.node_nums:
                raise ValueError(f"Time {t}: Node count {num_nodes} does not match network node count {self.node_nums}")
            
            # 初始化拓扑矩阵为全0，表示无连接
            cur_link_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
            
            # 遍历节点，计算节点之间的距离并判断是否在通信范围内
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:  # 节点与自身距离为0，不考虑
                        distance = np.linalg.norm(node_positions[i] - node_positions[j])
                        if distance <= config.ENV_PARAMS['communication_range']:
                            cur_link_matrix[i][j] = 1
            
            link_matrix_list.append(cur_link_matrix)
        
        return link_matrix_list

    def get_current_state(self):
        # 计算链路动态度
        end_time = self.cur_prd_start_time
        start_time = max(self.cur_prd_start_time - self.DQN_k, 0)
        
        # 检查是否有足够的历史数据
        if end_time - start_time != self.DQN_k:
            raise ValueError(f"Insufficient history data: got {end_time - start_time} time points, expected {self.DQN_k}")
        
        # link_matrix_list为历史self.DQN_k个拓扑
        link_matrix_list = self.get_link_matrix_list(start_time, end_time)
        
        # link_dynamic包含self.DQN_k-1个动态度
        link_dynamic = self.calculate_link_dynamics(link_matrix_list)
        
        # 检查计算得到的动态度维度是否正确
        if link_dynamic.shape != (self.DQN_k - 1, 3):
            raise ValueError(f"Unexpected link dynamics shape: {link_dynamic.shape}, expected ({self.DQN_k-1}, 3)")
        
        # 将链路动态度展平为一维数组
        link_dynamic_flat = link_dynamic.flatten()
        
        # 和上一轮的DyPrd值self.pre_DyPrd组合形成当前状态
        current_state_flat = np.concatenate([link_dynamic_flat, [self.pre_DyPrd]])
        
        return current_state_flat

    # 获取指定时间窗口内网络拓扑的动态变化信息，并将其扁平化为一维特征向量返回
    def get_p_t(self, current_time, Length):
        # 获取历史数据，self.data包含节点的历史数据
        history_data = self.data[(self.data['Time'] >= current_time - Length) & (self.data['Time'] <= current_time)]

        # 提取节点坐标信息
        node_positions = self.data[(self.data['Time'] == current_time)]
        # 初始化拓扑变化信息列表
        topology_changes_flat = []

        # 遍历节点数据，计算拓扑变化信息
        for index, row in node_positions.iterrows():
            node_id = row['Node']
            x = row['X']
            y = row['Y']
            z = row['Z']
            # 获取当前节点的历史位置信息
            node_history = history_data[history_data['Node'] == node_id][['Time', 'X', 'Y', 'Z']].values

            # 初始化新增和减少的邻居节点列表
            new_neighbors = np.zeros((self.node_nums), dtype=int)
            lost_neighbors = np.zeros((self.node_nums), dtype=int)

            # 获取当前时间节点的邻居节点列表
            current_neighbors = self.get_neighbors(node_id, node_positions)

            # 获取Time-Length时间前节点的邻居节点列表
            prev_time = current_time - Length
            prev_positions = self.data[(self.data['Time'] == prev_time)]
            prev_neighbors = self.get_neighbors(node_id, prev_positions)

            # 计算新增和减少的邻居节点
            for neighbor in current_neighbors:
                if neighbor not in prev_neighbors:
                    new_neighbors[int(neighbor)] = 1

            for neighbor in prev_neighbors:
                if neighbor not in current_neighbors:
                    lost_neighbors[int(neighbor)] = 1

            # 计算位置向量的变化
            position_changes = []
            for i in range(1, len(node_history)):
                position_change = node_history[i][1:] - node_history[i - 1][1:]
                position_changes += position_change.tolist()

            # 将节点拓扑特征 = [新增邻居向量] + [丢失邻居向量] + [位置变化序列]组合成一维列表
            node_topology_changes = new_neighbors.tolist() + lost_neighbors.tolist() + position_changes

            # 添加节点的拓扑变化信息到扁平化的总列表
            topology_changes_flat.extend(node_topology_changes)

        return topology_changes_flat

    def get_neighbors(self, node_id, current_positions):
        # 初始化邻居节点列表
        neighbors = []

        node_info = current_positions.loc[current_positions['Node'] == node_id]
        x1 = node_info['X'].values[0]
        y1 = node_info['Y'].values[0]
        z1 = node_info['Z'].values[0]
        # 遍历所有节点
        for index, row in current_positions.iterrows():
            other_node_id = row['Node']
            x2 = row['X']
            y2 = row['Y']
            z2 = row['Z']
            if other_node_id != node_id:
                # 计算节点之间的距离
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                # 如果距离在一定范围内，则将该节点视为邻居
                if distance <= config.ENV_PARAMS['communication_range']:
                    neighbors.append(other_node_id)

        return neighbors

    def predict_topology(self, DyPrd):
        if not self.lstm_models:
            self.load_lstm_model()

        predicted_trajectories = []
        end_time = self.cur_prd_start_time  # 以tb作为结束点
        start_time = max(end_time - 23, 0)  # 计算起始时间，确保时间窗口长度为24个时间单位
        
        for node_index in range(9):
            scaler = self.scalers_list[node_index]
            # 获得各个节点的未归一化实际轨迹数据，用于输入lstm进行拓扑推演
            node_specific_data = self.data[self.data['Node'] == node_index]
            # 从 node_specific_data（该节点的完整轨迹数据）选取从start_time到end_time的数据
            # 仅保留 'X', 'Y', 'Z' 列，生成一个形状为 (24, 3)的DataFrame
            node_data = node_specific_data[(node_specific_data['Time'] >= start_time) & (node_specific_data['Time'] <= end_time)][
                ['X', 'Y', 'Z']]
            # 打印node_data的形状
            # print(f"Shape of node_data: {node_data.shape}")

            input_sequence = scaler.transform(node_data.values) # 对输入数据进行归一化
            input_sequence = input_sequence.reshape(1, 24, 3)  # 输入维度为(1, 24, 3)
            input_tensor = torch.tensor(input_sequence, dtype=torch.float32).to(device)

            model = self.lstm_models[node_index]
            model.eval()

            with torch.no_grad():
                model_output = model(input_tensor)

            # 将模型输出转移到CPU，然后转换为NumPy数组
            model_output = model_output.cpu().numpy()  # 这里输出维度为(1, 12, 3)
            model_output_squeezed = np.squeeze(model_output, axis=0)  # 得到维度 (12, 3)            
            # 对预测结果进行反归一化
            predictions = scaler.inverse_transform(model_output_squeezed)
            if DyPrd > predictions.shape[0]:
                raise ValueError(f"DyPrd={DyPrd}超过LSTM预测步数{predictions.shape[0]}")    
                    
            predicted_position = predictions[0:DyPrd, :]  # 获取前DyPrd个预测结果
            # predicted_positions列表包含了9个节点的预测轨迹，每个轨迹是形状为 (DyPrd, 3)的数组
            predicted_trajectories.append(predicted_position)

        # 转换为NumPy数组，形状为(节点数, DyPrd, 3)
        predicted_trajectories = np.array(predicted_trajectories)  # 形状: (9, DyPrd, 3)
        
        # 计算拓扑矩阵数组
        num_nodes = len(predicted_trajectories)
        topology_matrices = []  # 存储每个时间点的拓扑矩阵
        
        for t in range(DyPrd):  # 遍历每个预测时间点
            topology_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        # 获取节点i和j在时间点t的位置
                        pos_i = predicted_trajectories[i, t, :]
                        pos_j = predicted_trajectories[j, t, :]
                        distance = np.linalg.norm(pos_i - pos_j)
                        
                        if distance <= config.ENV_PARAMS['communication_range']:
                            topology_matrix[i][j] = 1
            
            topology_matrices.append(topology_matrix)
        # 返回一个长度为 DyPrd的列表，其中每个元素是该预测时间点的 (9, 9)的邻接矩阵
        return topology_matrices

    def calculate_reward(self, slot_allocation_matrix):
        """
        先计算真实的拓扑和去除时隙冲突之后的时隙分配矩阵，然后根据这个时隙分配矩阵和真实的拓扑计算时延和吞吐量，最后计算奖励。
        :param slot_allocation_matrix:
        :param future_time:
        :return:
        """
        # 读取真实的拓扑
        rel_topology = self.get_cur_link_matrix_list()
        # slot_matrix为去除时隙冲突之后的时隙分配矩阵数组，包含DyPrd个无冲突矩阵
        # col_list为包含了DyPrd个冲突时隙数的数组
        slot_matrix, col_list = self.cal_collision(rel_topology, slot_allocation_matrix)
        slot_matrix_array = np.array(slot_matrix)
        print(f"去除冲突之后的时隙分配矩阵形状为{slot_matrix_array.shape}")
        self.coll = sum(col_list)
        
        N = self.node_nums  # 节点的数量
        C = 1  # 每个链路在一个时隙内可以传输的数据量，1代表吞吐量=时隙数
        DyPrd_len = len(slot_matrix)  # 获取时隙分配矩阵的数量

        # 初始化总时延和总吞吐量
        self.T_sum = 0

        # 遍历每个时隙分配矩阵
        for t in range(DyPrd_len):
            # 获取当前时间的时隙分配矩阵
            current_slot_matrix = slot_matrix[t]
            # 遍历每个节点计算吞吐量
            for j in range(N):
                # 计算单个节点的吞吐量
                T_j = sum(current_slot_matrix[j]) * C
                # 累加到总吞吐量
                self.T_sum += T_j
        
        # 或者使用更简洁的numpy方式：
        # self.T_sum = np.sum(slot_matrix_array) * C
        
        # 更新时隙分配方案占用的吞吐量
        boardcast_cost = config.ENV_PARAMS['boardcast_cost']
        # 计算奖励
        reward = self.slot_reward_ratio * (self.T_sum - boardcast_cost) / self.DyPrd - 0.3 * sum(self.diff) - self.coll

        return reward
    
    def cal_collision(self, rel_topology_list, slot_matrix):
        """
        验证时隙分配矩阵在真实的拓扑中是否有冲突，对每个时间点独立处理
        输入: rel_topology_list - 真实拓扑矩阵列表，长度为DyPrd
            slot_matrix - 基于预测拓扑生成的时隙分配矩阵
        输出: slot_matrix_list - 每个时间点处理后的时隙分配矩阵列表
            col_list - 每个时间点冲突时隙数的列表
        """
        slot_matrix_list = []
        col_list = []
        
        for rel_topy in rel_topology_list:
            # 创建时隙分配矩阵的副本，避免修改原始矩阵
            current_slot_matrix = slot_matrix.copy()
            col = 0
            
            # 计算二阶邻居
            neib_2 = one_two_neighbors(rel_topy)
            
            for ii in range(self.node_nums):
                for jj in range(ii + 1, self.node_nums):
                    if rel_topy[ii][jj] == 1 or neib_2[ii][jj] == 1:
                        for s in range(self.slot_nums):
                            if (current_slot_matrix[ii][s] + current_slot_matrix[jj][s] > 1):
                                # 如果两个节点在同一个时隙上都有分配，则存在冲突
                                col += 2
                                current_slot_matrix[ii][s] = 0
                                current_slot_matrix[jj][s] = 0
            
            slot_matrix_list.append(current_slot_matrix)
            col_list.append(col)
        
        return slot_matrix_list, col_list
    
    def save_slot(self, time, slot_matrix, path):
        """
        记录每个时间对应的时隙分配矩阵
        :param time:时间
        :param slot_matrix:时隙分配矩阵
        :param path: 保存路径
        """
        # 构建文件名，包含时间信息
        filename = f"slot_{time}.npy"

        # 保存时隙分配矩阵为.npy文件
        np.save(os.path.join(path, filename), slot_matrix)

    def get_diff(self, predicted_matrices, real_matrices):
        """
        比较两个邻接矩阵数组的差异
        输入: predicted_matrices - 预测拓扑列表，长度为DyPrd
            real_matrices - 实际拓扑列表，长度为DyPrd
        输出: diff_counts - 每个时间点差异数组成的数组，长度为DyPrd
        """
        # 确保两个数组长度相同
        if len(predicted_matrices) != len(real_matrices):
            raise ValueError(f"数组长度不一致: 预测{len(predicted_matrices)} vs 实际{len(real_matrices)}")
        
        diff_counts = np.zeros(len(predicted_matrices), dtype=int)
        
        for i in range(len(predicted_matrices)):
            # 获取当前时间点的预测和实际矩阵
            pre_matrix = predicted_matrices[i]
            rel_matrix = real_matrices[i]
            
            # 确保两个矩阵的形状相同
            if pre_matrix.shape != rel_matrix.shape:
                raise ValueError(f"矩阵{i}形状不一致: 预测{pre_matrix.shape} vs 实际{rel_matrix.shape}")
            
            # 计算两个矩阵的差异
            diff_matrix = np.logical_xor(pre_matrix, rel_matrix)
            
            # 统计不同的元素个数
            diff_counts[i] = np.sum(diff_matrix)
        
        return diff_counts


if __name__ == '__main__':
    pass