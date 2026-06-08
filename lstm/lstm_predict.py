import torch
import torch.nn as nn
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import LSTM model definition
from lstm_model import TopologyPredictor
from config.config import config

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 负责加载模型、执行预测和评估结果。
class LSTMPredictor:
    """LSTM Model Predictor"""
    
    def __init__(self, model_path, data_dir, node_id=0):
        """
        Initialize LSTM predictor
        
        Args:
            model_path: Model save path
            data_dir: Data directory path
            node_id: Node ID
        """
        self.node_id = node_id
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load data
        self.load_data()
        
        # Load scaler
        self.load_scaler()
        
        # Initialize model
        self.model = self.load_model()

    # 数据加载        
    def load_data(self):
        """Load test data"""
        data_path = os.path.join(self.data_dir, f"node_{self.node_id}_data.npz")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = np.load(data_path)
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        
        print(f"Test data loaded successfully")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        print(f"Number of test samples: {len(self.X_test)}")
    
    # 标准化器加载方法    
    def load_scaler(self):
        """Load scaler - using joblib"""
        scaler_path = os.path.join(self.data_dir, f"scaler_node_{self.node_id}.pkl")
        
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        # Load scaler using joblib
        self.scaler = joblib.load(scaler_path)
        
        print(f"Scaler loaded successfully, type: {type(self.scaler)}")
        
    def load_model(self):
        """Load LSTM model"""
        # Initialize model structure (parameters must match training)
        model = TopologyPredictor(
            input_size=config.LSTM_PARAMS['input_size'],  # 3
            hidden_size=config.LSTM_PARAMS.get('hidden_size', 128),
            num_layers=config.LSTM_PARAMS.get('num_layers', 2),
            pred_len=config.LSTM_PARAMS['pred_len']  # 12
        ).to(self.device)
        
        # Load model weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully: {self.model_path}")
        
        return model
    
    # 对输入批次进行前向传播预测
    def predict(self, X_batch):
        """Batch prediction"""
        with torch.no_grad():
            X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()
    
    # 反归一化
    def inverse_transform(self, data):
        """Inverse transform data"""
        # Note: Original data shape should be (n_samples, seq_len, features)
        original_shape = data.shape
        data_2d = data.reshape(-1, data.shape[-1])
        data_inv = self.scaler.inverse_transform(data_2d)
        return data_inv.reshape(original_shape)
    
    # 评估模型在测试集上的预测性能
    def evaluate_predictions(self, n_samples=8, save_trajectories=True, output_dir="./lstm/lstm_result", sample_indices=None):
        """Evaluate model prediction performance and save trajectory data
        
        Args:
            n_samples: Number of samples to evaluate
            save_trajectories: Whether to save trajectory data to file
            output_dir: Directory to save trajectory data files
            sample_indices: Specific sample indices to evaluate. If None, samples are randomly selected.
            
        Returns:
            y_true_original: Ground truth trajectories
            y_pred_original: Predicted trajectories
            errors: Error metrics
            traj_data_path: Path to saved trajectory data file (if saved)
            selected_indices: Indices of the selected samples
        """
        print(f"\nStarting model evaluation...")
        
        # Get total number of test samples
        total_samples = len(self.X_test)
        
        # Select sample indices
        if sample_indices is not None:
            # Use provided sample indices
            selected_indices = sample_indices
            n_samples = len(selected_indices)
            print(f"Using {n_samples} specified sample indices")
        elif n_samples < total_samples:
            # Randomly select n_samples from test set
            selected_indices = random.sample(range(total_samples), n_samples)
            print(f"Randomly selected {n_samples} samples from {total_samples} test samples")
        else:
            # Use all samples
            selected_indices = list(range(total_samples))
            n_samples = total_samples
            print(f"Using all {n_samples} test samples")
        
        # Take selected test samples
        X_eval = self.X_test[selected_indices]
        y_true = self.y_test[selected_indices]
        
        # Predict
        y_pred = self.predict(X_eval)
        
        # Inverse transform
        X_eval_original = self.inverse_transform(X_eval)
        y_true_original = self.inverse_transform(y_true)
        y_pred_original = self.inverse_transform(y_pred)
        
        # Save trajectory data to file
        traj_data_path = None
        if save_trajectories:
            traj_data_path = self.save_trajectory_data(
                X_eval_original, y_true_original, y_pred_original, 
                n_samples, output_dir, selected_indices
            )
        
        # Calculate errors
        errors = self.calculate_errors(y_true_original, y_pred_original)
        
        # Print statistics
        self.print_error_statistics(errors)
        
        # Plot error distribution
        self.plot_error_distribution(errors)
        
        return y_true_original, y_pred_original, errors, traj_data_path, selected_indices

    def save_trajectory_data(self, X_history, y_true, y_pred, n_samples, output_dir="./lstm/lstm_result", sample_indices=None):
        """Save trajectory data to JSON file
        
        Args:
            X_history: Historical trajectories (n_samples, 24, 3)
            y_true: Ground truth future trajectories (n_samples, 12, 3)
            y_pred: Predicted future trajectories (n_samples, 12, 3)
            n_samples: Number of samples
            output_dir: Directory to save the file
            sample_indices: Indices of the selected samples in the original dataset
            
        Returns:
            file_path: Path to the saved JSON file
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"node_{self.node_id}_trajectories.json"
        file_path = output_path / filename
        
        # Prepare data structure
        trajectory_data = {
            "node_id": int(self.node_id),
            "n_samples": int(n_samples),
            "sample_indices": sample_indices if sample_indices is not None else list(range(n_samples)),
            "samples": []
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for i in range(n_samples):
            sample_data = {
                "sample_id": int(i),
                "original_index": int(trajectory_data["sample_indices"][i]),
                "history": X_history[i].tolist(),  # 24x3
                "ground_truth": y_true[i].tolist(),  # 12x3
                "prediction": y_pred[i].tolist()    # 12x3
            }
            
            # Calculate per-sample errors for reference
            mse_x = np.mean((y_true[i, :, 0] - y_pred[i, :, 0]) ** 2)
            mse_y = np.mean((y_true[i, :, 1] - y_pred[i, :, 1]) ** 2)
            mse_z = np.mean((y_true[i, :, 2] - y_pred[i, :, 2]) ** 2)
            mse_total = np.mean((y_true[i] - y_pred[i]) ** 2)
            
            sample_data["errors"] = {
                "mse_x": float(mse_x),
                "mse_y": float(mse_y),
                "mse_z": float(mse_z),
                "mse_total": float(mse_total),
                "rmse_total": float(np.sqrt(mse_total))
            }
            
            trajectory_data["samples"].append(sample_data)
        
        # Calculate overall statistics
        all_mse = np.mean((y_true - y_pred) ** 2)
        trajectory_data["overall_statistics"] = {
            "mse_total": float(all_mse),
            "rmse_total": float(np.sqrt(all_mse)),
            "mse_x": float(np.mean((y_true[:, :, 0] - y_pred[:, :, 0]) ** 2)),
            "mse_y": float(np.mean((y_true[:, :, 1] - y_pred[:, :, 1]) ** 2)),
            "mse_z": float(np.mean((y_true[:, :, 2] - y_pred[:, :, 2]) ** 2)),
            "data_shape": {
                "history": f"{X_history.shape[0]}x{X_history.shape[1]}x{X_history.shape[2]}",
                "ground_truth": f"{y_true.shape[0]}x{y_true.shape[1]}x{y_true.shape[2]}",
                "prediction": f"{y_pred.shape[0]}x{y_pred.shape[1]}x{y_pred.shape[2]}"
            }
        }
        
        # Save to JSON file
        with open(file_path, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"\nTrajectory data saved to: {file_path}")
        print(f"Total samples saved: {n_samples}")
        if sample_indices is not None:
            print(f"Sample indices: {sample_indices}")
        print(f"Data structure: History={X_history.shape}, Ground Truth={y_true.shape}, Prediction={y_pred.shape}")
        
        return str(file_path)
    
    # 计算的误差指标：
    # 整体误差：MSE、MAE、RMSE
    # 各维度误差（X, Y, Z）：MSE、RMSE、MAE
    # 时间步误差：每个预测时间步的MSE
    # 返回：包含所有误差指标的字典
    def calculate_errors(self, y_true, y_pred):
        """Calculate various error metrics"""
        errors = {}
        
        # Ensure consistent shape
        n_samples, pred_len, n_features = y_true.shape
        
        # Calculate error for each sample
        mse_per_sample = np.zeros(n_samples)
        mae_per_sample = np.zeros(n_samples)
        rmse_per_sample = np.zeros(n_samples)
        
        for i in range(n_samples):
            mse_per_sample[i] = mean_squared_error(
                y_true[i].flatten(), 
                y_pred[i].flatten()
            )
            mae_per_sample[i] = mean_absolute_error(
                y_true[i].flatten(), 
                y_pred[i].flatten()
            )
            rmse_per_sample[i] = np.sqrt(mse_per_sample[i])
        
        # Overall errors
        errors['MSE'] = np.mean(mse_per_sample)
        errors['MAE'] = np.mean(mae_per_sample)
        errors['RMSE'] = np.mean(rmse_per_sample)
        
        # Dimension-wise errors
        for dim, dim_name in enumerate(['X', 'Y', 'Z']):
            mse_dim = mean_squared_error(
                y_true[:, :, dim].flatten(),
                y_pred[:, :, dim].flatten()
            )
            mae_dim = mean_absolute_error(
                y_true[:, :, dim].flatten(),
                y_pred[:, :, dim].flatten()
            )
            
            errors[f'MSE_{dim_name}'] = mse_dim
            errors[f'RMSE_{dim_name}'] = np.sqrt(mse_dim)
            errors[f'MAE_{dim_name}'] = mae_dim
        
        # Timestep-wise errors
        mse_per_timestep = np.zeros(pred_len)
        for t in range(pred_len):
            mse_per_timestep[t] = mean_squared_error(
                y_true[:, t, :].flatten(),
                y_pred[:, t, :].flatten()
            )
        errors['MSE_per_timestep'] = mse_per_timestep
        
        return errors
    
    def print_error_statistics(self, errors):
        """Print error statistics"""
        print("\n" + "="*60)
        print("Model Prediction Error Statistics")
        print("="*60)
        
        print(f"{'Metric':<20} {'Value':<20} {'Unit':<10}")
        print("-"*50)
        
        # Overall errors
        print(f"{'Average MSE':<20} {errors['MSE']:<20.6f} {'m²'}")
        print(f"{'Average MAE':<20} {errors['MAE']:<20.6f} {'m'}")
        print(f"{'Average RMSE':<20} {errors['RMSE']:<20.6f} {'m'}")
        
        print("-"*50)
        
        # Dimension-wise errors
        for dim, dim_name in enumerate(['X', 'Y', 'Z']):
            print(f"{f'{dim_name} Dimension MSE':<20} {errors[f'MSE_{dim_name}']:<20.6f} {'m²'}")
            print(f"{f'{dim_name} Dimension RMSE':<20} {errors[f'RMSE_{dim_name}']:<20.6f} {'m'}")
            print(f"{f'{dim_name} Dimension MAE':<20} {errors[f'MAE_{dim_name}']:<20.6f} {'m'}")
            print("-"*50)
        
        # Timestep-wise errors
        print("\nPrediction Errors by Timestep:")
        mse_per_timestep = errors['MSE_per_timestep']
        for t in range(len(mse_per_timestep)):
            print(f"  Timestep {t+1:2d}: MSE = {mse_per_timestep[t]:.6f} m², "
                  f"RMSE = {np.sqrt(mse_per_timestep[t]):.6f} m")
    
    # 生成误差分布的可视化图表 node_{node_id}_error_analysis.png
    def plot_error_distribution(self, errors):
        """Plot error distribution"""
        # Create error dataframe
        error_data = {
            'MSE': errors['MSE'],
            'MAE': errors['MAE'],
            'RMSE': errors['RMSE'],
            'MSE_X': errors['MSE_X'],
            'MSE_Y': errors['MSE_Y'],
            'MSE_Z': errors['MSE_Z']
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Node {self.node_id} Model Prediction Error Distribution', fontsize=16, fontweight='bold')
        
        # Overall error bar chart
        overall_errors = ['MSE', 'MAE', 'RMSE']
        overall_values = [errors[e] for e in overall_errors]
        axes[0, 0].bar(overall_errors, overall_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Overall Error Metrics')
        axes[0, 0].set_ylabel('Error Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(overall_values):
            axes[0, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        # Dimension-wise MSE bar chart
        dim_errors = ['MSE_X', 'MSE_Y', 'MSE_Z']
        dim_values = [errors[e] for e in dim_errors]
        axes[0, 1].bar(['X', 'Y', 'Z'], dim_values, color=['red', 'green', 'blue'])
        axes[0, 1].set_title('MSE by Dimension')
        axes[0, 1].set_ylabel('MSE')
        axes[0, 1].grid(True, alpha=0.3)
        
        for i, v in enumerate(dim_values):
            axes[0, 1].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        # Dimension-wise RMSE bar chart
        rmse_dim_values = [errors[f'RMSE_{dim}'] for dim in ['X', 'Y', 'Z']]
        axes[0, 2].bar(['X', 'Y', 'Z'], rmse_dim_values, color=['red', 'green', 'blue'])
        axes[0, 2].set_title('RMSE by Dimension')
        axes[0, 2].set_ylabel('RMSE (m)')
        axes[0, 2].grid(True, alpha=0.3)
        
        for i, v in enumerate(rmse_dim_values):
            axes[0, 2].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        # Dimension-wise MAE bar chart
        mae_dim_values = [errors[f'MAE_{dim}'] for dim in ['X', 'Y', 'Z']]
        axes[1, 0].bar(['X', 'Y', 'Z'], mae_dim_values, color=['red', 'green', 'blue'])
        axes[1, 0].set_title('MAE by Dimension')
        axes[1, 0].set_ylabel('MAE (m)')
        axes[1, 0].grid(True, alpha=0.3)
        
        for i, v in enumerate(mae_dim_values):
            axes[1, 0].text(i, v, f'{v:.4f}', ha='center', va='bottom')
        
        # Timestep-wise error line chart
        mse_per_timestep = errors['MSE_per_timestep']
        rmse_per_timestep = np.sqrt(mse_per_timestep)
        axes[1, 1].plot(range(1, len(mse_per_timestep)+1), mse_per_timestep, 
                       marker='o', color='purple', linewidth=2)
        axes[1, 1].set_title('MSE by Timestep')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(range(1, len(rmse_per_timestep)+1), rmse_per_timestep,
                       marker='s', color='orange', linewidth=2)
        axes[1, 2].set_title('RMSE by Timestep')
        axes[1, 2].set_xlabel('Timestep')
        axes[1, 2].set_ylabel('RMSE (m)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'node_{self.node_id}_error_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nError analysis plot saved as: node_{self.node_id}_error_analysis.png")
        plt.show()
    
    # n_samples：要可视化的样本数量
    # sample_indices：指定要可视化的样本索引列表
    def visualize_predictions(self, n_samples=8, sample_indices=None):
        """Visualize prediction results
        
        Args:
            n_samples: Number of samples to visualize
            sample_indices: Specific sample indices to visualize. If None, randomly select n_samples.
        """
        print(f"\nStarting prediction visualization...")
        
        # If no specific indices provided, evaluate predictions first
        if sample_indices is None:
            # Evaluate random samples
            y_true_original, y_pred_original, _, _, sample_indices = self.evaluate_predictions(
                n_samples=n_samples, 
                save_trajectories=False
            )
        else:
            # Use specified sample indices
            y_true_original, y_pred_original, _, _, _ = self.evaluate_predictions(
                n_samples=len(sample_indices),
                sample_indices=sample_indices,
                save_trajectories=False
            )
        
        # Create visualization for each sample
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx >= len(self.X_test):
                print(f"Warning: Sample index {sample_idx} out of range, skipping")
                continue
                
            self.plot_single_prediction(sample_idx, y_true_original, y_pred_original, i+1)
        
        # Create 3D trajectory plot
        # self.plot_3d_trajectories(sample_indices[:min(3, len(sample_indices))], y_true_original, y_pred_original)
    
    # 绘制单个样本的轨迹预测图
    # 生成内容：
    # 三个子图分别显示X、Y、Z坐标的历史轨迹、真实未来轨迹和预测轨迹
    # 每个子图包含误差统计信息
    # 历史与未来的分界线
    # 输出文件：node_{node_id}_prediction_sample{sample_idx}_plot{plot_idx}.png
    def plot_single_prediction_output12(self, sample_idx, y_true_original, y_pred_original, plot_idx):
        """Plot prediction results for a single sample with improved layout"""
        # Get X_test for current sample (historical data)
        X_sample = self.X_test[sample_idx]
        X_sample_original = self.inverse_transform(X_sample.reshape(1, -1, 3))[0]
        
        # Get true and predicted values
        y_true_sample = y_true_original[plot_idx-1]
        y_pred_sample = y_pred_original[plot_idx-1]
        
        # Time points
        history_time = np.arange(1, 25)
        future_time = np.arange(25, 37)
        
        # Create figure with adjusted aspect ratio
        fig, axes = plt.subplots(3, 1, figsize=(13, 10))
        fig.suptitle(f'Node Trajectory Prediction', 
                    fontsize=16, fontweight='bold')
        
        # Adjust subplot spacing
        plt.subplots_adjust(hspace=0.35)
        
        # Axis labels
        dimensions = ['X Coordinate (m)', 'Y Coordinate (m)', 'Z Coordinate (m)']
        
        for dim in range(3):
            ax = axes[dim]
            
            # Plot historical trajectory
            ax.plot(history_time, X_sample_original[:, dim], 
                   color='blue', linewidth=2, marker='o', markersize=4,
                   label='Historical')
            
            # Plot future true trajectory
            ax.plot(future_time, y_true_sample[:, dim],
                   color='green', linewidth=2, marker='s', markersize=4,
                   label='Actual')
            
            # Plot future predicted trajectory
            ax.plot(future_time, y_pred_sample[:, dim],
                   color='red', linewidth=2, linestyle='--', marker='^', markersize=4,
                   label='Predicted')
            
            # Calculate and display errors
            mse = mean_squared_error(y_true_sample[:, dim], y_pred_sample[:, dim])
            mae = mean_absolute_error(y_true_sample[:, dim], y_pred_sample[:, dim])
            
            # Add error information
            error_text = f'MSE: {mse:.4f}  MAE: {mae:.4f}'
            ax.text(0.02, 0.95, error_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set plot properties
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel(dimensions[dim], fontsize=11)
            
            # Move legend to avoid overlap - placed outside the plot
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                     borderaxespad=0., fontsize=10, frameon=True)
            
            ax.grid(True, alpha=0.3)
            
            # Set x-axis ticks
            ax.set_xticks(np.arange(0, 37, 5))
            ax.set_xlim(0, 37)
            
            # # Add history/future dividing line
            # ax.axvline(x=24.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            # ax.text(12, ax.get_ylim()[1] * 0.9, 'History', ha='center', fontsize=10, fontweight='bold')
            # ax.text(30, ax.get_ylim()[1] * 0.9, 'Future', ha='center', fontsize=10, fontweight='bold')
            
            # Increase tick label size
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Adjust layout to make room for legends
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Left padding to accommodate legends
        
        # Save the figure
        plt.savefig(f'node_{self.node_id}_prediction_sample{sample_idx}_plot{plot_idx}.png', 
                   dpi=300, bbox_inches='tight')
        print(f'生成图片node_{self.node_id}_prediction_sample{sample_idx}_plot{plot_idx}.png')
        plt.show()
        
        print(f"Trajectory prediction plot for sample {sample_idx} saved")
        
    def plot_single_prediction(self, sample_idx, y_true_original, y_pred_original, plot_idx):
        """Plot prediction results for a single sample with improved layout
        """
        # Get X_test for current sample (historical data)
        X_sample = self.X_test[sample_idx]
        X_sample_original = self.inverse_transform(X_sample.reshape(1, -1, 3))[0]
        
        # Get true and predicted values
        y_true_sample = y_true_original[plot_idx-1]
        y_pred_sample = y_pred_original[plot_idx-1]
        
        # ========== 关键修改1：调整未来时间步范围为25~30 ==========
        history_time = np.arange(1, 25)
        future_time = np.arange(25, 31)  # arange左闭右开，31才能包含30
        
        # Create figure with adjusted aspect ratio
        fig, axes = plt.subplots(3, 1, figsize=(13, 10))
        fig.suptitle(f'Node Trajectory Prediction', 
                    fontsize=16, fontweight='bold')
        
        # Adjust subplot spacing
        plt.subplots_adjust(hspace=0.35)
        
        # Axis labels
        dimensions = ['X Coordinate (m)', 'Y Coordinate (m)', 'Z Coordinate (m)']
        
        for dim in range(3):
            ax = axes[dim]
            
            # Plot historical trajectory
            ax.plot(history_time, X_sample_original[:, dim], 
                color='blue', linewidth=2, marker='o', markersize=4,
                label='Historical')
            
            # ========== 关键修改2：仅取前6个未来点（25~30） ==========
            # Plot future true trajectory (仅展示25~30)
            ax.plot(future_time, y_true_sample[:6, dim],  # 切片取前6个点
                color='green', linewidth=2, marker='s', markersize=4,
                label='Actual')
            
            # Plot future predicted trajectory (仅展示25~30)
            ax.plot(future_time, y_pred_sample[:6, dim],  # 切片取前6个点
                color='red', linewidth=2, linestyle='--', marker='^', markersize=4,
                label='Predicted')
            
            # ========== 关键修改3：删除MSE/MAE计算和显示代码 ==========
            
            # Set plot properties
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel(dimensions[dim], fontsize=11)
            
            # Move legend to avoid overlap - placed outside the plot
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                    borderaxespad=0., fontsize=10, frameon=True)
            
            ax.grid(True, alpha=0.3)
            
            # ========== 优化：调整x轴刻度和范围，适配25~30的展示 ==========
            ax.set_xticks(np.arange(0, 32, 5))  # 刻度调整为0~31，步长5
            ax.set_xlim(0, 31)  # x轴范围适配25~30
            
            # Increase tick label size
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Adjust layout to make room for legends
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Left padding to accommodate legends
        
        # Save the figure
        plt.savefig(f'node_{self.node_id}_prediction_sample{sample_idx}_plot{plot_idx}.png', 
                dpi=300, bbox_inches='tight')
        print(f'生成图片node_{self.node_id}_prediction_sample{sample_idx}_plot{plot_idx}.png')
        plt.show()
        
        print(f"Trajectory prediction plot for sample {sample_idx} saved")
    
    # 生成最多3个样本的3D轨迹图 node_{node_id}_3d_trajectories.png
    def plot_3d_trajectories(self, sample_indices, y_true_original, y_pred_original):
        """Plot 3D trajectories (up to 3 samples)"""
        if len(sample_indices) > 3:
            sample_indices = sample_indices[:3]
        
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(f'Node {self.node_id} - 3D Trajectory Visualization', 
                    fontsize=16, fontweight='bold')
        
        for i, sample_idx in enumerate(sample_indices):
            if sample_idx >= len(self.X_test):
                continue
                
            # Get historical data
            X_sample = self.X_test[sample_idx]
            X_sample_original = self.inverse_transform(X_sample.reshape(1, -1, 3))[0]
            
            # Get true and predicted values
            y_true_sample = y_true_original[i]  # Use i as index since y_true_original is already filtered
            y_pred_sample = y_pred_original[i]
            
            # Create 3D subplot
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            
            # Plot historical trajectory
            ax.plot(X_sample_original[:, 0], X_sample_original[:, 1], X_sample_original[:, 2],
                   color='blue', linewidth=2, marker='o', markersize=4, label='History')
            
            # Plot future true trajectory
            ax.plot(y_true_sample[:, 0], y_true_sample[:, 1], y_true_sample[:, 2],
                   color='green', linewidth=2, marker='s', markersize=4, label='Future Actual')
            
            # Plot future predicted trajectory
            ax.plot(y_pred_sample[:, 0], y_pred_sample[:, 1], y_pred_sample[:, 2],
                   color='red', linewidth=2, linestyle='--', marker='^', markersize=4, label='Future Predicted')
            
            # Connect last point of historical trajectory to first points of future trajectories
            last_history = X_sample_original[-1]
            first_true = y_true_sample[0]
            first_pred = y_pred_sample[0]
            
            ax.plot([last_history[0], first_true[0]], 
                   [last_history[1], first_true[1]],
                   [last_history[2], first_true[2]],
                   color='green', linestyle=':', linewidth=1, alpha=0.7)
            
            ax.plot([last_history[0], first_pred[0]], 
                   [last_history[1], first_pred[1]],
                   [last_history[2], first_pred[2]],
                   color='red', linestyle=':', linewidth=1, alpha=0.7)
            
            # Set plot properties
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Sample {sample_idx} (Index {sample_idx})')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'node_{self.node_id}_3d_trajectories.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"3D trajectory plot saved as: node_{self.node_id}_3d_trajectories.png")

def main():
    """Main function"""
    print("="*60)
    print("LSTM Trajectory Prediction Model Testing")
    print("="*60)

    # Set paths
    data_dir = "/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/traj_data/processed_data/"
    model_path = "/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/lstm/saved_models/model_node_0.pth"
    node_id = 0
    
    # Check if paths exist
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Create predictor
    print(f"\nInitializing LSTM predictor (Node {node_id})...")
    predictor = LSTMPredictor(model_path, data_dir, node_id)
    
    # Test model performance with random samples
    print("\n" + "="*60)
    print("Testing Model Performance (Random Samples)")
    print("="*60)
    
    # Option 1: Use default random selection
    y_true, y_pred, errors, traj_data_path, selected_indices = predictor.evaluate_predictions(
        n_samples=8, 
        save_trajectories=True,
        output_dir="./lstm/lstm_result"
    )
    
    print(f"\nSelected sample indices: {selected_indices}")
    
    # Option 2: You can also specify your own sample indices
    # specific_indices = [0, 5, 10, 15, 20, 25, 30, 35]
    # y_true, y_pred, errors, traj_data_path, selected_indices = predictor.evaluate_predictions(
    #     n_samples=8, 
    #     save_trajectories=True,
    #     output_dir="./lstm/lstm_result",
    #     sample_indices=specific_indices
    # )
    
    # Visualize prediction results
    print("\n" + "="*60)
    print("Visualizing Prediction Results")
    print("="*60)
    
    # Option 1: Use randomly selected samples (from evaluation)
    #predictor.visualize_predictions(sample_indices=selected_indices)
    
    # Option 2: Randomly select different samples for visualization
    predictor.visualize_predictions(n_samples=8)
    
    # Option 3: Specify specific sample indices for visualization
    # specific_indices = [0, 5, 10, 15, 20, 25, 30, 35]
    # predictor.visualize_predictions(sample_indices=specific_indices)
    
    print("\n" + "="*60)
    print("Testing completed!")
    print(f"Trajectory data saved to: {traj_data_path}")
    print("="*60)

if __name__ == "__main__":
    main()