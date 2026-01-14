import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import joblib
import argparse
import traceback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 设置英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'

# 基础的编码器-解码器LSTM模型，用于序列到序列的预测
class Seq2SeqModel(nn.Module):
    """三维轨迹预测Seq2Seq模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_steps):
        super(Seq2SeqModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        # 编码器LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 解码器LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 输出层（预测三维坐标）
        self.fc = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        # 编码输入序列
        _, (hidden, cell) = self.encoder(x)
        
        # 准备解码器输入（初始化为零）
        decoder_input = torch.zeros(x.size(0), 1, self.hidden_size).to(x.device)
        
        # 存储输出序列
        outputs = []
        
        # 逐步解码
        for _ in range(self.output_steps):
            # 解码一步
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            
            # 预测三维坐标
            pred = self.fc(out.squeeze(1))
            outputs.append(pred)
            
            # 使用当前预测作为下一步输入
            decoder_input = out
        
        # 组合所有预测结果 [batch_size, output_steps, 3]
        return torch.stack(outputs, dim=1)
     
# 加载之前训练时保存的数据归一化器, 确保预测时的数据归一化与训练时一致
def load_scaler(node_id, scaler_dir):
    """加载归一化器"""
    scaler_path = os.path.join(scaler_dir, f'scaler_node_{node_id}.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"归一化器未找到: {scaler_path}")
    return joblib.load(scaler_path)

# 加载预训练的LSTM模型权重，将模型设置为评估模式
def load_model(node_id, model_dir, output_steps, hidden_size=128, num_layers=2):
    """加载预训练模型"""
    model = Seq2SeqModel(
        input_size=3, 
        hidden_size=hidden_size, 
        num_layers=num_layers,
        output_steps=output_steps
    )
    
    model_path = os.path.join(model_dir, f"model_node_{node_id}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型未找到: {model_path}")
    
    # 根据设备加载模型
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()
    return model

# 在整个测试集上评估模型性能，计算整个测试集的MSE和MAE指标，将评估结果保存到文本文件
def evaluate_on_test_set(node_id, data_dir, model_dir, scaler_dir, seq_len=24, pred_steps=12):
    """在整个测试集上评估模型性能"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 从.npz文件加载预处理的测试数据
        data_path = os.path.join(data_dir, f"node_{node_id}_data.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"节点 {node_id} 的数据文件不存在: {data_path}")
        
        data = np.load(data_path)
        
        # 只加载测试集
        X_test = data['X_test']
        y_test = data['y_test']
        
        print(f"加载测试集数据: X_test形状={X_test.shape}, y_test形状={y_test.shape}")
        
        # 加载归一化器
        scaler = load_scaler(node_id, scaler_dir)
        
        # 加载模型
        model = load_model(node_id, model_dir, pred_steps)
        model.to(device)
        
        # 转换为Tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        
        # 批量预测
        batch_size = 32
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                batch_x = X_test_tensor[i:i+batch_size]
                predictions = model(batch_x)
                all_predictions.append(predictions.cpu().numpy())
        
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        # 反归一化
        predictions_denorm = []
        y_test_denorm = []
        
        for i in range(len(all_predictions)):
            # 反归一化预测
            pred_denorm = scaler.inverse_transform(all_predictions[i])
            predictions_denorm.append(pred_denorm)
            
            # 反归一化真实值
            true_denorm = scaler.inverse_transform(y_test[i])
            y_test_denorm.append(true_denorm)
        
        predictions_denorm = np.array(predictions_denorm)
        y_test_denorm = np.array(y_test_denorm)
        
        # 计算整体误差指标
        mse_total = mean_squared_error(
            y_test_denorm.reshape(-1, 3), 
            predictions_denorm.reshape(-1, 3)
        )
        mae_total = mean_absolute_error(
            y_test_denorm.reshape(-1, 3), 
            predictions_denorm.reshape(-1, 3)
        )
        
        print(f"\n{'='*50}")
        print(f"节点 {node_id} 测试集评估结果:")
        print(f"测试样本数: {len(X_test)}")
        print(f"总MSE: {mse_total:.6f}")
        print(f"总MAE: {mae_total:.6f}")
        print(f"{'='*50}")
        
        # 保存评估结果
        output_dir = "test_evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, f"node_{node_id}_evaluation.txt"), 'w') as f:
            f.write(f"节点 {node_id} 测试集评估结果:\n")
            f.write(f"测试样本数: {len(X_test)}\n")
            f.write(f"总MSE: {mse_total:.6f}\n")
            f.write(f"总MAE: {mae_total:.6f}\n")
        
        return predictions_denorm, y_test_denorm, mse_total, mae_total
        
    except Exception as e:
        print(f"节点 {node_id} 测试集评估失败: {str(e)}")
        traceback.print_exc()
        return None, None, None, None

# 从测试集中选择多个样本进行可视化，对每个样本展示三维轨迹+三个坐标分量，计算每个样本的独立误差指标
def predict_test_samples(node_id, data_dir, model_dir, scaler_dir, 
                         seq_len=24, pred_steps=12, num_samples=4):
    """可视化测试集中的部分样本预测结果"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载数据
        data_path = os.path.join(data_dir, f"node_{node_id}_data.npz")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"节点 {node_id} 的数据文件不存在: {data_path}")
        
        data = np.load(data_path)
        
        # 加载测试集
        X_test = data['X_test']
        y_test = data['y_test']
        
        # 只取前几个样本用于可视化
        num_samples = min(num_samples, len(X_test))
        X_vis = X_test[:num_samples]
        y_vis = y_test[:num_samples]
        
        print(f"可视化测试集中的 {num_samples} 个样本")
        
        # 加载归一化器
        scaler = load_scaler(node_id, scaler_dir)
        
        # 加载模型
        model = load_model(node_id, model_dir, pred_steps)
        model.to(device)
        
        # 反归一化输入
        X_vis_denorm = []
        for i in range(num_samples):
            x_denorm = scaler.inverse_transform(X_vis[i])
            X_vis_denorm.append(x_denorm)
        X_vis_denorm = np.array(X_vis_denorm)
        
        # 转换为Tensor
        X_vis_tensor = torch.tensor(X_vis, dtype=torch.float32).to(device)
        
        # 进行预测
        with torch.no_grad():
            predictions = model(X_vis_tensor)
        
        # 反归一化预测结果
        predictions_np = predictions.cpu().numpy()
        predictions_denorm = []
        for i in range(num_samples):
            pred_denorm = scaler.inverse_transform(predictions_np[i])
            predictions_denorm.append(pred_denorm)
        predictions_denorm = np.array(predictions_denorm)
        
        # 反归一化真实值
        y_vis_denorm = []
        for i in range(num_samples):
            y_denorm = scaler.inverse_transform(y_vis[i])
            y_vis_denorm.append(y_denorm)
        y_vis_denorm = np.array(y_vis_denorm)
        
        # 创建可视化图表
        fig = plt.figure(figsize=(20, 5*num_samples))
        fig.suptitle(f'Node {node_id} - Test Set Predictions (First {num_samples} Samples)', 
                     fontsize=20, y=0.98)
        
        for i in range(num_samples):
            # 创建时间序列
            history_times = np.arange(1, seq_len + 1)
            future_times = np.arange(seq_len + 1, seq_len + pred_steps + 1)
            
            # 计算单个样本的误差
            sample_mse = mean_squared_error(y_vis_denorm[i], predictions_denorm[i])
            sample_mae = mean_absolute_error(y_vis_denorm[i], predictions_denorm[i])
            
            print(f"样本 {i+1}: MSE={sample_mse:.6f}, MAE={sample_mae:.6f}")
            
            # 1. 三维轨迹图
            ax1 = fig.add_subplot(num_samples, 4, i*4+1, projection='3d')
            
            # 历史轨迹 (蓝色)
            ax1.plot(X_vis_denorm[i, :, 0], X_vis_denorm[i, :, 1], X_vis_denorm[i, :, 2], 
                     'b-', linewidth=2, label='History')
            
            # 真实未来轨迹 (绿色)
            ax1.plot(y_vis_denorm[i, :, 0], y_vis_denorm[i, :, 1], y_vis_denorm[i, :, 2],
                     'g-', linewidth=2, label='True Future')
            
            # 预测轨迹 (红色虚线)
            ax1.plot(predictions_denorm[i, :, 0], predictions_denorm[i, :, 1], predictions_denorm[i, :, 2],
                     'r--', linewidth=2, label='Prediction')
            
            # 连接点
            ax1.scatter(X_vis_denorm[i, -1, 0], X_vis_denorm[i, -1, 1], X_vis_denorm[i, -1, 2],
                       s=100, c='purple', marker='o', label='Transition Point')
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title(f"Sample {i+1} - 3D Trajectory\nMSE: {sample_mse:.4f}, MAE: {sample_mae:.4f}")
            ax1.legend()
            
            # 2. X坐标分量
            ax2 = fig.add_subplot(num_samples, 4, i*4+2)
            ax2.plot(history_times, X_vis_denorm[i, :, 0], 'b-', label='History', linewidth=2)
            ax2.plot(future_times, y_vis_denorm[i, :, 0], 'g-', label='True Future', linewidth=2)
            ax2.plot(future_times, predictions_denorm[i, :, 0], 'r--', label='Prediction', linewidth=2)
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('X (m)')
            ax2.set_title(f"Sample {i+1} - X Coordinate")
            ax2.legend()
            ax2.grid(True)
            
            # 3. Y坐标分量
            ax3 = fig.add_subplot(num_samples, 4, i*4+3)
            ax3.plot(history_times, X_vis_denorm[i, :, 1], 'b-', label='History', linewidth=2)
            ax3.plot(future_times, y_vis_denorm[i, :, 1], 'g-', label='True Future', linewidth=2)
            ax3.plot(future_times, predictions_denorm[i, :, 1], 'r--', label='Prediction', linewidth=2)
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Y (m)')
            ax3.set_title(f"Sample {i+1} - Y Coordinate")
            ax3.legend()
            ax3.grid(True)
            
            # 4. Z坐标分量
            ax4 = fig.add_subplot(num_samples, 4, i*4+4)
            ax4.plot(history_times, X_vis_denorm[i, :, 2], 'b-', label='History', linewidth=2)
            ax4.plot(future_times, y_vis_denorm[i, :, 2], 'g-', label='True Future', linewidth=2)
            ax4.plot(future_times, predictions_denorm[i, :, 2], 'r--', label='Prediction', linewidth=2)
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Z (m)')
            ax4.set_title(f"Sample {i+1} - Z Coordinate")
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 保存结果
        output_dir = "test_predictions"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"node_{node_id}_test_predictions.png"), dpi=300, bbox_inches='tight')
        print(f"测试集预测结果已保存至 {output_dir}/node_{node_id}_test_predictions.png")
        
        plt.show()
        
        return predictions_denorm, y_vis_denorm
        
    except Exception as e:
        print(f"节点 {node_id} 测试集可视化失败: {str(e)}")
        traceback.print_exc()
        return None, None

def evaluate_top_samples_with_details(node_id, data_dir, model_dir, scaler_dir, 
                                     seq_len=24, pred_steps=12, num_samples=300):
    """评估前num_samples个测试样本，并记录详细预测结果"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载数据
        data_path = os.path.join(data_dir, f"node_{node_id}_data.npz")
        data = np.load(data_path)
        
        # 只取前num_samples个样本
        X_test = data['X_test'][:num_samples]  # 形状: (num_samples, 24, 3)
        y_test = data['y_test'][:num_samples]  # 形状: (num_samples, 12, 3)
        
        print(f"评估前 {num_samples} 个测试样本")
        print(f"X_test形状: {X_test.shape}")
        print(f"y_test形状: {y_test.shape}")
        
        # 加载归一化器和模型
        scaler = load_scaler(node_id, scaler_dir)
        model = load_model(node_id, model_dir, pred_steps)
        model.to(device)
        
        # 转换为Tensor
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        # 批量预测
        batch_size = 32
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X_test_tensor), batch_size):
                batch_x = X_test_tensor[i:i+batch_size]
                predictions = model(batch_x)
                all_predictions.append(predictions.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)  # (num_samples, 12, 3)
        
        # 反归一化
        predictions_denorm = []
        y_test_denorm = []
        
        for i in range(num_samples):
            # 预测值反归一化
            pred_denorm = scaler.inverse_transform(predictions[i])
            predictions_denorm.append(pred_denorm)
            
            # 真实值反归一化
            true_denorm = scaler.inverse_transform(y_test[i])
            y_test_denorm.append(true_denorm)
        
        predictions_denorm = np.array(predictions_denorm)  # (num_samples, 12, 3)
        y_test_denorm = np.array(y_test_denorm)  # (num_samples, 12, 3)
        
        # 计算误差指标
        mse_total = mean_squared_error(
            y_test_denorm.reshape(-1, 3), 
            predictions_denorm.reshape(-1, 3)
        )
        mae_total = mean_absolute_error(
            y_test_denorm.reshape(-1, 3), 
            predictions_denorm.reshape(-1, 3)
        )
        
        print(f"\n前 {num_samples} 个样本评估结果:")
        print(f"MSE: {mse_total:.6f}")
        print(f"MAE: {mae_total:.6f}")
        print(f"RMSE: {np.sqrt(mse_total):.6f}")
        print(f"总预测点数: {num_samples * pred_steps} 个点")
        
        # 保存预测结果
        output_dir = "detailed_predictions"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为.npz文件
        np.savez(
            os.path.join(output_dir, f"node_{node_id}_top{num_samples}_detailed.npz"),
            predictions=predictions_denorm,
            ground_truth=y_test_denorm
        )
        
        # 创建详细的CSV文件
        create_detailed_csv(predictions_denorm, y_test_denorm, node_id, num_samples, pred_steps, output_dir)
        
        # 计算每个样本的误差
        sample_errors = []
        for i in range(num_samples):
            sample_mse = mean_squared_error(y_test_denorm[i], predictions_denorm[i])
            sample_mae = mean_absolute_error(y_test_denorm[i], predictions_denorm[i])
            sample_rmse = np.sqrt(sample_mse)
            sample_errors.append({
                'sample_id': i,
                'mse': sample_mse,
                'mae': sample_mae,
                'rmse': sample_rmse
            })
        
        # 保存每个样本的误差统计
        error_df = pd.DataFrame(sample_errors)
        error_df.to_csv(os.path.join(output_dir, f"node_{node_id}_top{num_samples}_error_summary.csv"), index=False)
        
        # 误差分布分析
        analyze_error_distribution_details(predictions_denorm, y_test_denorm, node_id, num_samples, pred_steps, output_dir)
        
        return predictions_denorm, y_test_denorm, mse_total, mae_total
        
    except Exception as e:
        print(f"评估前{num_samples}个样本失败: {str(e)}")
        traceback.print_exc()
        return None, None, None, None

def create_detailed_csv(predictions, ground_truth, node_id, num_samples, pred_steps, output_dir):
    """创建详细预测结果的CSV文件"""
    
    # 准备数据列表
    data_rows = []
    
    for sample_id in range(num_samples):
        for step in range(pred_steps):
            # 获取真实值和预测值
            true_x, true_y, true_z = ground_truth[sample_id, step]
            pred_x, pred_y, pred_z = predictions[sample_id, step]
            
            # 计算误差
            error_x = true_x - pred_x
            error_y = true_y - pred_y
            error_z = true_z - pred_z
            euclidean_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
            
            data_rows.append({
                'sample_id': sample_id,
                'step': step + 1,  # 从1开始计数
                'true_x': true_x,
                'true_y': true_y,
                'true_z': true_z,
                'pred_x': pred_x,
                'pred_y': pred_y,
                'pred_z': pred_z,
                'error_x': error_x,
                'error_y': error_y,
                'error_z': error_z,
                'euclidean_error': euclidean_error
            })
    
    # 创建DataFrame
    df = pd.DataFrame(data_rows)
    
    # 重新排列列顺序
    df = df[['sample_id', 'step', 
             'true_x', 'true_y', 'true_z',
             'pred_x', 'pred_y', 'pred_z',
             'error_x', 'error_y', 'error_z',
             'euclidean_error']]
    
    # 保存CSV
    csv_path = os.path.join(output_dir, f"node_{node_id}_top{num_samples}_detailed_predictions.csv")
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"详细预测结果已保存到: {csv_path}")
    print(f"文件包含 {len(df)} 行数据（{num_samples}个样本 × {pred_steps}步）")
    
    # 保存汇总统计
    summary_stats(df, node_id, num_samples, pred_steps, output_dir)
    
    return df

def summary_stats(df, node_id, num_samples, pred_steps, output_dir):
    """计算并保存汇总统计"""
    
    # 按样本ID分组计算统计
    sample_stats = df.groupby('sample_id').agg({
        'euclidean_error': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # 重命名列
    sample_stats.columns = ['avg_error', 'std_error', 'min_error', 'max_error']
    sample_stats = sample_stats.reset_index()
    
    # 按时间步分组计算统计
    step_stats = df.groupby('step').agg({
        'euclidean_error': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # 重命名列
    step_stats.columns = ['avg_error', 'std_error', 'min_error', 'max_error']
    step_stats = step_stats.reset_index()
    
    # 保存统计结果
    sample_stats_path = os.path.join(output_dir, f"node_{node_id}_top{num_samples}_sample_stats.csv")
    step_stats_path = os.path.join(output_dir, f"node_{node_id}_top{num_samples}_step_stats.csv")
    
    sample_stats.to_csv(sample_stats_path, index=False)
    step_stats.to_csv(step_stats_path, index=False)
    
    print(f"样本统计已保存到: {sample_stats_path}")
    print(f"时间步统计已保存到: {step_stats_path}")
    
    # 打印整体统计
    print(f"\n整体统计:")
    print(f"平均欧氏距离误差: {df['euclidean_error'].mean():.4f} 米")
    print(f"误差标准差: {df['euclidean_error'].std():.4f} 米")
    print(f"最小误差: {df['euclidean_error'].min():.4f} 米")
    print(f"最大误差: {df['euclidean_error'].max():.4f} 米")
    print(f"中位数误差: {df['euclidean_error'].median():.4f} 米")
    
    # 误差分布统计
    print(f"\n误差分布:")
    thresholds = [1, 2, 5, 10, 20, 50]
    for threshold in thresholds:
        percentage = (df['euclidean_error'] < threshold).mean() * 100
        print(f"误差 < {threshold} 米: {percentage:.1f}%")

def analyze_error_distribution_details(predictions, ground_truth, node_id, num_samples, pred_steps, output_dir):
    """分析误差分布（使用英文）"""
    # 计算欧氏距离
    errors = np.sqrt(np.sum((predictions - ground_truth)**2, axis=2))  # (num_samples, pred_steps)
    
    print(f"\nError Distribution Statistics ({num_samples} samples):")
    print(f"Minimum Error: {np.min(errors):.2f} m")
    print(f"Maximum Error: {np.max(errors):.2f} m")
    print(f"Mean Error: {np.mean(errors):.2f} m")
    print(f"Median Error: {np.median(errors):.2f} m")
    print(f"Error Std: {np.std(errors):.2f} m")
    print(f"25th Percentile: {np.percentile(errors, 25):.2f} m")
    print(f"75th Percentile: {np.percentile(errors, 75):.2f} m")
    print(f"90th Percentile: {np.percentile(errors, 90):.2f} m")
    
    # 可视化误差分布
    plt.figure(figsize=(12, 8))
    
    # 1. 误差直方图
    plt.subplot(2, 2, 1)
    plt.hist(errors.flatten(), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Error (m)')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution (Node {node_id}, {num_samples} samples)')
    plt.grid(True, alpha=0.3)
    
    # 2. 误差随预测步数的变化
    plt.subplot(2, 2, 2)
    time_errors = np.mean(errors, axis=0)  # 每个时间步的平均误差
    time_std = np.std(errors, axis=0)
    plt.plot(range(1, pred_steps+1), time_errors, 'b-o', linewidth=2)
    plt.fill_between(range(1, pred_steps+1), 
                     time_errors - time_std,
                     time_errors + time_std,
                     alpha=0.3, color='b')
    plt.xlabel('Prediction Step')
    plt.ylabel('Mean Error (m)')
    plt.title('Error vs Prediction Step')
    plt.grid(True, alpha=0.3)
    
    # 3. 坐标分量误差
    plt.subplot(2, 2, 3)
    component_errors = []
    component_names = ['X', 'Y', 'Z']
    for i in range(3):
        comp_error = np.abs(predictions[:, :, i] - ground_truth[:, :, i]).flatten()
        component_errors.append(comp_error)
    
    box_data = [component_errors[0], component_errors[1], component_errors[2]]
    plt.boxplot(box_data, labels=component_names)
    plt.xlabel('Coordinate')
    plt.ylabel('Absolute Error (m)')
    plt.title('Coordinate-wise Error Distribution')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. 累积分布函数
    plt.subplot(2, 2, 4)
    sorted_errors = np.sort(errors.flatten())
    cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    plt.plot(sorted_errors, cdf, 'b-', linewidth=2)
    plt.xlabel('Error (m)')
    plt.ylabel('Cumulative Probability')
    plt.title('Error Cumulative Distribution Function')
    plt.grid(True, alpha=0.3)
    
    # 添加一些参考线
    for threshold in [1, 5, 10, 20]:
        if threshold < sorted_errors[-1]:
            idx = np.searchsorted(sorted_errors, threshold)
            plt.axvline(x=threshold, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
            plt.text(threshold+0.5, 0.9, f'{cdf[idx]:.2%}', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"node_{node_id}_error_analysis.png"), dpi=300, bbox_inches='tight')
    print(f"误差分析图已保存到: {output_dir}/node_{node_id}_error_analysis.png")
    plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='轨迹预测 - 测试集评估')
    parser.add_argument('--data_dir', type=str, default='processed_data', help='处理后的数据目录')
    parser.add_argument('--model_dir', type=str, default='trained_models', help='模型目录')
    parser.add_argument('--scaler_dir', type=str, default='processed_data', help='归一化器目录')
    parser.add_argument('--node_id', type=int, required=True, help='要预测的节点ID')
    parser.add_argument('--seq_len', type=int, default=24, help='历史序列长度')
    parser.add_argument('--pred_steps', type=int, default=12, help='预测步数')
    parser.add_argument('--num_samples', type=int, default=4, help='可视化样本数量')
    parser.add_argument('--evaluate_only', action='store_true', help='仅评估不可视化')
    parser.add_argument('--detailed_eval', action='store_true', help='执行详细评估并保存CSV')
    
    args = parser.parse_args()
    
    # 验证目录存在
    for dir_path in [args.data_dir, args.model_dir, args.scaler_dir]:
        if not os.path.exists(dir_path):
            print(f"错误: 目录不存在 - {dir_path}")
            return
    
    # 执行详细评估
    if args.detailed_eval:
        print("="*60)
        print(f"执行详细评估 - 节点 {args.node_id}")
        print(f"将评估前 {args.num_samples} 个样本")
        print("="*60)
        
        predictions, true_values, mse_total, mae_total = evaluate_top_samples_with_details(
            node_id=args.node_id,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            scaler_dir=args.scaler_dir,
            seq_len=args.seq_len,
            pred_steps=args.pred_steps,
            num_samples=args.num_samples
        )
        
        if predictions is not None:
            print(f"\n详细评估完成！")
            print(f"结果保存在 'detailed_predictions/' 目录中")

if __name__ == "__main__":
    main()