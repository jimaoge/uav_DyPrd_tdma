# uav_DyPrd_tdma/config/config.py
import os
from pathlib import Path

class UAVConfig:
    """UAV网络TDMA时隙分配系统配置"""
    
    def __init__(self):
        # ---------- 环境参数 ----------
        self.ENV_PARAMS = {
            'uav_nums': 9,         # UAV数量
            'communication_range': 300, # 通信范围(米)
            'slot_nums': 9,        # 时隙数量
            'boardcast_cost': 3     # 每轮周期所需的更新时隙分配方案的开销时隙数
        }
        # ---------- 基础路径 ----------
        # 获取项目根目录（config.py所在目录的父目录）
        self.PROJECT_ROOT = Path(__file__).parent.parent
        
        # ---------- 数据路径 ----------
        self.DATA_PATHS = {
            'origin_traj_data': self.PROJECT_ROOT / "traj_data" / "origin_data" ,
            'processed_traj_data': self.PROJECT_ROOT / "traj_data" / "processed_data",
            'saved_lstm_models_path': self.PROJECT_ROOT / "lstm" / "saved_models",
            'saved_scaler_path' : self.PROJECT_ROOT / "traj_data" / "processed_data",
            'saved_dqn_models_path' : self.PROJECT_ROOT/"DRL"/"saved_models",  # 保存完成训练的dqn模型
            'traj_data_file_name' : "node_position_1s_5s_5_0.7_int.csv",   # 轨迹文件名称
            'saved_slots_matrix_path': self.PROJECT_ROOT /"ga"/"saved_slots_matrix",  # 保存的时隙矩阵文件所在路径
            'saved_population_path' : self.PROJECT_ROOT /"ga"/"saved_populations",  # 保存的种群文件所在路径
            'saved_population_file_name' : 'population.pkl',  # 保存的种群文件名
            'DQN_train_result_path': self.PROJECT_ROOT / "results" / "DQN_training" ,  # 保存DQN训练结果
            'DQN_test_result_path': self.PROJECT_ROOT / "results" / "DQN_testing" ,  # 保存DQN测试结果
        }
        
        # ---------- DQN参数（DyPrd决策） ----------        
        DQN_k = 6
        STATE_DIM = 3 * (DQN_k - 1) + 1
        self.DQN_PARAMS = {
            'STATE_DIM': STATE_DIM,     # 策略网络和目标网络输入的长度:状态空间的维度
            'ACTION_DIM': 10,           # 策略网络和目标网络输出的长度:动作空间的维度
            'BATCH_SIZE': 128,          # batch_size
            'MEMORY_CAPACITY': 6000,    # 经验回放的容量
            'TARGET_UPDATE': 400,       # target网络更新的频率
            'GAMMA': 0.9,               # 回报折扣率
            'LR': 0.002,                # 学习率
            'DQN_k': DQN_k,             # DQN需要输入历史k - 1个链路动态性
            'max_time': 15000,          # DQN训练数据的最大处理时间范围
            'max_steps_per_episode': 100,  # 每个episode包含几轮周期/几次DyPrd决策
            'total_episode_in_train': 500,  # 一次训练包含多少个episode
            'DQN_train_save_interval': 10,  # 每多少个episode保存一次指标
            'slot_reward_ratio': 1.0,       # 计算奖励时，时隙吞吐量的占比
            'epsilon_start': 0.9,           # 随机探索率的初始值
            'epsilon_end' : 0.05,           # 随机探索率的最小值
            'epsilon_decay' : 0.985,        # 衰减率
        }
     
        # ---------- LSTM数据预处理参数 ----------
        self.PREPROCESS_PARAMS = {
            'seq_len': 24,          # 输入序列长度 (历史轨迹点数)
            'pred_steps': 12,       # 输出序列长度 (预测轨迹点数)
            'train_ratio': 0.6,     # 训练集比例
            'val_ratio': 0.2,       # 验证集比例
            'test_ratio': 0.2,      # 测试集比例
            'max_time': 15000       # 最大处理时间范围
        }
                
        # ---------- LSTM拓扑预测参数 ----------
        self.LSTM_PARAMS = {
            'seq_len': 24,          # 输入序列长度 (历史轨迹点数)
            'pred_steps': 12,       # 输出序列长度 (预测轨迹点数)
            'epochs': 60,           
            'batch_size': 32,   # 批量大小，每次训练时输入模型的样本数
            'input_size': 3,      # xyz位置
            'hidden_size': 128,   # LSTM隐藏层维度，决定模型记忆和表达能力
            'num_layers': 2,      # LSTM堆叠层数，增加网络深度，层数越多，模型表示能力越强
            'pred_len': 12,       # 预测序列长度，即要预测的未来时间步数
            'seq_len': 24,        # 输入序列长度，历史观测时间步数
            'patience': 0,        # 验证损失连续几次未改善，提前停止训练。0为不会提前停止
            'dropout': 0.1,        # 每个mini-batch都会随机丢弃dropout比例的神经元，使网络不依赖于任何特定的神经元，增强了鲁棒性
        }
        
        # ---------- 遗传算法参数 ----------
        self.GA_PARAMS = {
            'N^2':  81,  # N的平方
            'population_size' : 100,  # 种群的个体数量
            'probability_of_cross' : 0.6,  # 交叉概率
            'probability_of_mutate' : 0.05,  # 变异概率
            'number_of_generation' : 300,  # 主算法循环次数
            'number_of_node' : 9,
            'number_of_slot' : 9
        }
        

# 创建全局配置实例
config = UAVConfig()