# uav_DyPrd_tdma/traj_data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import sys
from pathlib import Path

# 添加项目根目录到Python路径，以便导入config模块
sys.path.append(str(Path(__file__).parent.parent))
from config.config import config

def preprocess_multi_node_data(file_path, seq_len=24, pred_steps=12, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, max_time=15000):
    """
    预处理包含多个节点的轨迹数据文件（仅处理时间范围0~max_time的数据）
    将数据分为训练集、验证集和测试集三部分
    :param file_path: 轨迹文件路径
    :param seq_len: 历史序列长度
    :param pred_steps: 预测步数
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :param test_ratio: 测试集比例
    :param max_time: 最大处理时间范围
    :return: 处理后的数据集字典和归一化器字典
    """
    # 验证比例参数
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")
    
    # 读取数据
    print(f"开始处理文件: {file_path}")
    df = pd.read_csv(file_path, comment='#')
    
    # 只保留时间在0~max_time范围内的数据
    df = df[df['Time'] <= max_time]
    print(f"只处理时间范围0~{max_time}的数据，样本数: {len(df)}")
    
    # 检查数据质量
    if df.isnull().sum().sum() > 0:
        print(f"发现缺失值, 处理中...")
        df = df.dropna()
    
    # 按节点分组处理
    datasets = {}
    scalers = {}
    
    # 提取所有节点ID
    node_ids = df['Node'].unique()
    print(f"找到 {len(node_ids)} 个节点: {sorted(node_ids)}")
    
    for node_id in sorted(node_ids):
        print(f"\n处理节点 {node_id} 数据...")
        
        # 提取单个节点的数据并按时间排序
        node_df = df[df['Node'] == node_id].sort_values('Time')
        positions = node_df[['X', 'Y', 'Z']].values
        
        print(f"节点 {node_id} 总样本数: {len(positions)}")
        
        # 数据归一化 (使用MinMaxScaler范围(-1,1))
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_positions = scaler.fit_transform(positions)
        
        # 创建序列样本
        X, y = [], []
        for i in range(len(scaled_positions) - seq_len - pred_steps + 1):
            X.append(scaled_positions[i:i+seq_len])
            y.append(scaled_positions[i+seq_len:i+seq_len+pred_steps])
        
        if not X or not y:
            print(f"节点 {node_id} 数据不足，无法创建样本序列")
            continue
        
        X = np.array(X)
        y = np.array(y)
        
        # 数据集划分 (按时间顺序划分，保持时间连续性)
        total_samples = len(X)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        # 训练集
        X_train, y_train = X[:train_end], y[:train_end]
        # 验证集
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        # 测试集
        X_test, y_test = X[val_end:], y[val_end:]
        
        # 检查划分结果
        print(f"训练集: {len(X_train)} 样本 ({train_ratio*100:.0f}%)")
        print(f"验证集: {len(X_val)} 样本 ({val_ratio*100:.0f}%)")
        print(f"测试集: {len(X_test)} 样本 ({test_ratio*100:.0f}%)")
        
        datasets[node_id] = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        scalers[node_id] = scaler
        
        print(f"节点 {node_id} 数据准备完成")
    
    return datasets, scalers

def save_datasets(datasets, scalers, output_dir):
    """保存处理后的数据集"""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 保存归一化器
    for node_id, scaler in scalers.items():
        scaler_path = output_dir / f'scaler_node_{node_id}.pkl'
        joblib.dump(scaler, scaler_path)
    
    # 保存数据集
    for node_id, data in datasets.items():
        npz_path = output_dir / f'node_{node_id}_data.npz'
        np.savez_compressed(
            str(npz_path),  # 转换为字符串以兼容numpy
            X_train=data['train'][0],
            y_train=data['train'][1],
            X_val=data['val'][0],
            y_val=data['val'][1],
            X_test=data['test'][0],
            y_test=data['test'][1]
        )
        print(f"保存节点 {node_id} 数据到: {npz_path}")
    
    print(f"所有数据集已保存至 {output_dir}")

def main():
    """主处理函数"""
    # 获取配置文件中的路径
    origin_data_path = config.DATA_PATHS['origin_traj_data']
    processed_data_path = config.DATA_PATHS['processed_traj_data']
    data_file = config.DATA_PATHS['traj_data_file_name']

    file_path = origin_data_path / data_file
    
    # 检查文件是否存在
    if not file_path.exists():
        print(f"错误: 数据文件不存在 {file_path}")
        return
    
    #  从config.py获取预处理参数
    preprocess_params = config.PREPROCESS_PARAMS
    seq_len = preprocess_params['seq_len']        # 输入序列长度 (历史轨迹点数)
    pred_steps = preprocess_params['pred_steps']  # 输出序列长度 (预测轨迹点数)
    train_ratio = preprocess_params['train_ratio']  # 训练集比例
    val_ratio = preprocess_params['val_ratio']    # 验证集比例
    test_ratio = preprocess_params['test_ratio']  # 测试集比例
    max_time = preprocess_params['max_time']      # 最大处理时间范围
    
    print("使用以下预处理参数:")
    print(f"  seq_len: {seq_len} (历史轨迹点数)")
    print(f"  pred_steps: {pred_steps} (预测轨迹点数)")
    print(f"  train_ratio: {train_ratio} (训练集比例)")
    print(f"  val_ratio: {val_ratio} (验证集比例)")
    print(f"  test_ratio: {test_ratio} (测试集比例)")
    print(f"  max_time: {max_time} (最大处理时间)")
    
    # 处理数据
    datasets, scalers = preprocess_multi_node_data(
        str(file_path),  # 转换为字符串以兼容pandas
        seq_len=seq_len, 
        pred_steps=pred_steps, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        max_time=max_time
    )
    
    if datasets and scalers:
        save_datasets(datasets, scalers, processed_data_path)
        print("\n预处理完成! 数据格式已准备好用于模型训练")
        
        # 验证数据形状
        sample_node = list(datasets.keys())[0]
        X_train = datasets[sample_node]['train'][0]
        y_train = datasets[sample_node]['train'][1]
        X_val = datasets[sample_node]['val'][0]
        y_val = datasets[sample_node]['val'][1]
        X_test = datasets[sample_node]['test'][0]
        y_test = datasets[sample_node]['test'][1]
        
        print(f"\n节点 {sample_node} 数据形状验证:")
        print(f"训练集 - X_train 形状: {X_train.shape} (应为 [样本数, {seq_len}, 3])")
        print(f"训练集 - y_train 形状: {y_train.shape} (应为 [样本数, {pred_steps}, 3])")
        print(f"验证集 - X_val 形状: {X_val.shape} (应为 [样本数, {seq_len}, 3])")
        print(f"验证集 - y_val 形状: {y_val.shape} (应为 [样本数, {pred_steps}, 3])")
        print(f"测试集 - X_test 形状: {X_test.shape} (应为 [样本数, {seq_len}, 3])")
        print(f"测试集 - y_test 形状: {y_test.shape} (应为 [样本数, {pred_steps}, 3])")
        
        # 显示数据集大小比例
        total_samples = len(X_train) + len(X_val) + len(X_test)
        print(f"\n数据集分布:")
        print(f"训练集: {len(X_train)} 样本 ({len(X_train)/total_samples*100:.1f}%)")
        print(f"验证集: {len(X_val)} 样本 ({len(X_val)/total_samples*100:.1f}%)")
        print(f"测试集: {len(X_test)} 样本 ({len(X_test)/total_samples*100:.1f}%)")
    else:
        print("错误: 未生成有效数据集")

if __name__ == "__main__":
    main()