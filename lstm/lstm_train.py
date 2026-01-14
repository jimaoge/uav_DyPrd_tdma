import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import argparse
import sys
import re
from lstm_model import TopologyPredictor
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入配置
from config.config import config

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 数据集类
class TrajectoryDataset(Dataset):
    """轨迹数据集类"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PhysicsConstrainedLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1):
        super().__init__()
        self.position_loss = nn.MSELoss()
        self.alpha = alpha  # 速度平滑系数
        self.beta = beta    # 加速度约束系数
        
    def forward(self, pred, target):
        # 基本位置损失
        base_loss = self.position_loss(pred, target)
        
        # 速度平滑约束
        pred_vel = pred[:, 1:] - pred[:, :-1]
        targ_vel = target[:, 1:] - target[:, :-1]
        vel_loss = self.position_loss(pred_vel, targ_vel)
        
        # 加速度物理约束（无人机加速度有限）
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
        targ_acc = targ_vel[:, 1:] - targ_vel[:, :-1]
        acc_loss = torch.clamp(torch.abs(pred_acc) - 15.0, min=0).mean()  # 假设最大加速度15m/s²
        
        return base_loss + self.alpha * vel_loss + self.beta * acc_loss

# 训练函数
def train_model(node_id, data_dir, model_dir, epochs=100, batch_size=32, hidden_size=128, num_layers=2):
    """训练单个节点的模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    data_path = os.path.join(data_dir, f"node_{node_id}_data.npz")
    if not os.path.exists(data_path):
        print(f"节点 {node_id} 的数据文件不存在: {data_path}")
        return None
    
    data = np.load(data_path)
    
    # 只加载训练集和验证集，不加载测试集
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # 打印数据形状信息
    print(f"节点 {node_id} 训练数据形状: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"节点 {node_id} 验证数据形状: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # 创建数据集
    train_dataset = TrajectoryDataset(X_train, y_train)
    val_dataset = TrajectoryDataset(X_val, y_val)  # 修改2: 使用验证集代替原来的测试集
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 修改3: 验证集不需要打乱
    
    # 初始化模型
    model = TopologyPredictor(
        input_size=config.LSTM_PARAMS['input_size'],  #3
        hidden_size=hidden_size,
        num_layers=num_layers,
        pred_len=config.LSTM_PARAMS['pred_len'] #12
    ).to(device)
    
    print("="*50)
    print(f"训练节点 {node_id} 的设备确认:")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
    else:
        print("设备名称: CPU")
    print(f"模型已加载到: {next(model.parameters()).device}")
    print("="*50)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 训练记录
    train_losses = []
    val_losses = []
    
    patience = config.LSTM_PARAMS['patience']  # 连续8次验证损失不改善就停止
    early_stopping_counter = 0
    best_val_loss = float('inf')    
    
    print(f"\n训练节点 {node_id} 的模型...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:  # 使用验证集加载器
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        
        # 计算平均损失
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)  #使用验证集加载器长度
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(model_dir, f"model_node_{node_id}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"保存节点 {node_id} 的最佳模型，验证损失: {best_val_loss:.6f}")
            early_stopping_counter = 0  # 重置计数器
        else:
            early_stopping_counter += 1
        
        if patience > 0:
            if early_stopping_counter >= patience:
                print(f"验证损失连续 {patience} 次未改善，提前停止训练")
                break
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")
    
    # 保存训练历史
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Node {node_id} Training History')
    plt.legend()
    history_path = os.path.join(model_dir, f"training_history_node_{node_id}.png")
    plt.savefig(history_path)
    print(f"训练历史已保存至 {history_path}")
    
    return model

# 主训练函数
def train_all_nodes(data_dir, model_dir, epochs=100, batch_size=32, specific_node=None):
    """训练所有节点的模型或指定单个节点"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 获取预处理目录中的所有文件
    all_files = os.listdir(data_dir)
    print(f"目录 {data_dir} 中的文件列表: {all_files}")
    
    # 只处理有效的节点数据文件
    valid_nodes = []
    for filename in all_files:
        # 检查文件名格式: node_<数字>_data.npz
        if filename.startswith("node_") and filename.endswith("_data.npz"):
            # 提取节点ID部分
            parts = filename.split('_')
            if len(parts) >= 3:
                node_id = parts[1]
                
                # 验证节点ID是否为数字
                if node_id.isdigit():
                    # 检查文件是否实际存在
                    file_path = os.path.join(data_dir, filename)
                    if os.path.exists(file_path):
                        valid_nodes.append(node_id)
    
    # 确保没有重复节点
    valid_nodes = list(set(valid_nodes))
    
    # 按节点ID数字排序
    try:
        valid_nodes = sorted(valid_nodes, key=int)
    except:
        valid_nodes = sorted(valid_nodes)
    
    print(f"找到 {len(valid_nodes)} 个有效节点")
    print(f"节点列表: {valid_nodes}")
    
    if not valid_nodes:
        print("没有找到任何有效节点，训练中止")
        return
    
    # 如果指定了特定节点
    if specific_node is not None:
        specific_node = str(specific_node)
        if specific_node in valid_nodes:
            print(f"训练指定节点 {specific_node}")
            train_model(specific_node, data_dir, model_dir, epochs, batch_size)
        else:
            print(f"错误: 节点 {specific_node} 不存在，可用的节点: {valid_nodes}")
        return
    
    print(f"开始训练 {len(valid_nodes)} 个节点的模型...")
    
    for node_id in valid_nodes:
        train_model(node_id, data_dir, model_dir, epochs, batch_size)
    
    print("所有节点模型训练完成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练轨迹预测模型')
    # 使用config中的路径作为默认值
    parser.add_argument('--data_dir', type=str, default=str(config.DATA_PATHS['processed_traj_data']), 
                       help='预处理数据目录')
    parser.add_argument('--model_dir', type=str, default=str(config.DATA_PATHS['saved_lstm_models_path']), 
                       help='模型保存目录')
    parser.add_argument('--epochs', type=int, 
                       default=config.LSTM_PARAMS['epochs'], 
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, 
                       default=config.LSTM_PARAMS['batch_size'], 
                       help='批大小')
    parser.add_argument('--node_id', type=int, help='指定要训练的单个节点ID')
    args = parser.parse_args()
    
    # 训练节点模型
    train_all_nodes(
        data_dir=args.data_dir, 
        model_dir=args.model_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size,
        specific_node=args.node_id
    )