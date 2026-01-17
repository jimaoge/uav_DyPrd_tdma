# DQN网络模型,包括经验回访池的实现

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from random import choice, randrange, sample

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入配置
from config.config import config

# DQN模型参数
STATE_DIM = config.DQN_PARAMS['STATE_DIM']    # 策略网络和目标网络输入的长度:状态空间的维度
ACTION_DIM = config.DQN_PARAMS['ACTION_DIM']  # 策略网络和目标网络输出的长度:动作空间的维度
BATCH_SIZE = config.DQN_PARAMS['BATCH_SIZE']  # batch_size
MEMORY_CAPACITY = config.DQN_PARAMS['MEMORY_CAPACITY']  # 经验回放的容量
TARGET_UPDATE = config.DQN_PARAMS['TARGET_UPDATE']  # target网络更新的频率
GAMMA = config.DQN_PARAMS['GAMMA']  # 回报折扣率
LR = config.DQN_PARAMS['LR']  # 学习率
MODEL_PATH = config.DATA_PATHS['saved_dqn_models_path']  # DQN深度神经网络参数的保存路径

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义一个命名元组Transition，用于存储经验回放中的单个转移
# 包含：当前状态state、执行的动作action、下一个状态state_next、获得的奖励reward
Transition = namedtuple('Transition', ('state', 'action', 'state_next', 'reward'))

# 深度Q网络(DQN)的神经网络模型
class DNN(nn.Module):
    def __init__(self, n_state=STATE_DIM, n_action=ACTION_DIM):
        """
        初始化DQN网络结构
        参数:
            n_state: 状态空间的维度
            n_action: 动作空间的维度
        """
        super(DNN, self).__init__()
        # 输入层：状态维度 -> 1000个神经元
        self.input_layer = nn.Linear(n_state, 1000)
        self.input_layer.weight.data.normal_(0, 0.1)  # 用正态分布初始化权重
        
        # 隐藏层1：1000 -> 500个神经元
        self.middle_layer = nn.Linear(1000, 500)
        self.middle_layer.weight.data.normal_(0, 0.1)
        
        # 隐藏层2：500 -> 250个神经元
        self.middle_layer_2 = nn.Linear(500, 250)
        self.middle_layer_2.weight.data.normal_(0, 0.1)
        
        # 输出层（优势函数层）：250 -> 动作维度
        self.adv_layer = nn.Linear(250, n_action)
        self.adv_layer.weight.data.normal_(0, 0.1)

    def forward(self, state):
        """
        前向传播过程
        参数:
            state: 输入状态
        返回:
            out: 每个动作对应的Q值
        """
        x = F.relu(self.input_layer(state))  # 第一层 + ReLU激活
        x = F.relu(self.middle_layer(x))      # 第二层 + ReLU激活
        x = F.relu(self.middle_layer_2(x))    # 第三层 + ReLU激活
        out = self.adv_layer(x)               # 输出层（无激活函数）
        return out


# DQN代理类，包含经验回放、训练等功能
class DQN(object):
    def __init__(self):
        """
        初始化DQN代理
        包含策略网络、目标网络、经验回放池等组件
        """
        # 创建策略网络和目标网络
        self.policy_net = self.creat_model().to(device)  # 用于选择动作的网络
        self.target_net = self.creat_model().to(device)  # 用于计算目标Q值的网络
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 初始时使目标网络与策略网络相同
        
        # 经验回放池
        self.replay_size = MEMORY_CAPACITY  # 经验回放池的最大容量
        self.replay_queue = deque(maxlen=self.replay_size)  # 使用deque实现固定大小的队列
        
        # 训练相关参数
        self.learn_step_counter = 0  # 训练步数计数器
        self.loss_func = nn.MSELoss()  # 使用均方误差损失函数
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)  # 使用Adam优化器

    def creat_model(self):
        """创建DQN神经网络模型"""
        print("creating DNN.....")
        return DNN()

    def choose_action(self, s, epsilon):
        """
        使用ε-greedy策略选择动作
        参数:
            s: 当前状态
            epsilon: ε值，控制探索与利用的平衡
        返回:
            action: 选择的动作索引
        """
        # 以ε的概率随机选择动作（探索）
        if np.random.uniform() < epsilon:
            return np.random.choice(ACTION_DIM)
        # 以1-ε的概率选择当前估计最优的动作（利用）
        else:
            s = torch.FloatTensor(s).unsqueeze(0)  # 将状态转换为tensor，并添加批次维度
            if next(self.policy_net.parameters()).is_cuda:
                s = s.cuda()  # 如果模型在GPU上，将状态也移到GPU上

            with torch.no_grad():  # 禁用梯度计算，因为只是前向传播选择动作
                q_values = self.policy_net(s)  # 获取所有动作的Q值
                action = torch.argmax(q_values).item()  # 选择Q值最大的动作
            return action

    def remember(self, s, a, next_s, reward):
        """
        将状态转移存储到经验回放池
        参数:
            s: 当前状态
            a: 执行的动作
            next_s: 下一个状态
            reward: 获得的奖励
        """
        self.replay_queue.append(Transition(s, a, next_s, reward))

    def train(self):
        """训练DQN网络"""
        # 如果经验回放池不够大，不进行训练
        if len(self.replay_queue) < BATCH_SIZE:
            print(f"经验回放池不够大，当前长度：{len(self.replay_queue)}/{BATCH_SIZE}")
            return

        self.learn_step_counter += 1
        
        # 从经验回放池中随机采样一个批次
        transitions = self.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))  # 重组为批次的Transition

        # ========== 核心修复+优化：张量转换部分（原警告位置） ==========
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.long, device=device).unsqueeze(-1)
        reward_batch = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=device).unsqueeze(-1)
        state_next_batch = torch.tensor(np.array(batch.state_next), dtype=torch.float32, device=device)

        # 计算当前状态的Q值
        # gather(1, action_batch)选择执行动作对应的Q值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 使用目标网络计算下一个状态的最大Q值
        # detach()断开计算图，防止梯度传播到目标网络
        next_state_values = self.target_net(state_next_batch).max(1)[0].detach()

        # 计算目标Q值：reward + γ * max(Q(next_state))
        expected_state_action_values = reward_batch + (GAMMA * next_state_values).unsqueeze(1)

        # 计算损失：当前Q值与目标Q值的均方误差
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # 反向传播和优化
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0) # 梯度裁剪，max_norm可根据需求调1~5
        self.optimizer.step()  # 更新网络参数

        # 定期打印训练信息
        if self.learn_step_counter % 50 == 0:
            print(f"DQN训练迭代 {self.learn_step_counter}，损失：{loss.item()}")

        # 定期更新目标网络（将策略网络的参数复制到目标网络）
        if self.learn_step_counter % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def sample(self, batch_size):
        """
        从经验回放池中随机采样指定大小的样本
        参数:
            batch_size: 采样批次大小
        返回:
            随机采样的经验转移
        """
        return sample(self.replay_queue, batch_size)
    
    def save_model(self, model_dir=MODEL_PATH, filename="dqn_model.pth"):
        """
        保存DQN模型参数
        Args:
            model_dir: 保存模型文件的目录路径
            filename: 模型文件名，默认"dqn_model.pth"
        """
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / filename
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, model_path)
        
        print(f"模型已保存到: {model_path}")
        
    def load_model(self, model_dir=MODEL_PATH, filename="dqn_model.pth"):
        """
        加载DQN模型参数
        Args:
            model_dir: 模型文件所在的目录路径
            filename: 模型文件名，默认"dqn_model.pth"
        """
        # 构建完整的文件路径
        model_path = model_dir / filename
        
        # 加载模型
        checkpoint = torch.load(model_path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        
        # 只有当优化器状态存在时才加载
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"模型从 {model_path} 加载成功")