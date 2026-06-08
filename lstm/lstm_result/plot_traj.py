import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_trajectory_data(filepath):
    """
    从JSON文件加载轨迹数据
    
    参数:
    filepath: JSON文件路径
    
    返回:
    dict: 包含所有样本数据的字典
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # 检查数据结构
    print(f"节点ID: {data['node_id']}")
    print(f"样本数量: {data['n_samples']}")
    
    # 提取样本数据
    samples = data['samples']
    print(f"实际加载样本数: {len(samples)}")
    
    return samples

def plot_3d_trajectory(sample, sample_id, save_path=None):
    """
    绘制单个样本的3D轨迹图
    
    参数:
    sample: 单个样本数据
    sample_id: 样本ID
    save_path: 保存路径，如果为None则不保存
    """
    # 提取数据
    history = np.array(sample['history'])
    ground_truth = np.array(sample['ground_truth'])
    prediction = np.array(sample['prediction'])
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制历史轨迹
    ax.plot(history[:, 0], history[:, 1], history[:, 2], 
            label='History (24 steps)', color='blue', 
            marker='o', markersize=3, linewidth=2)
    
    # 绘制实际轨迹
    ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], 
            label='Ground Truth (12 steps)', color='green', 
            marker='s', markersize=4, linewidth=2)
    
    # 绘制预测轨迹
    ax.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], 
            label='Prediction (12 steps)', color='red', 
            marker='^', markersize=4, linewidth=2)
    
    # 设置坐标轴标签
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate', fontsize=12)
    
    # 设置标题
    ax.set_title(f'3D Trajectory - Sample {sample_id}', fontsize=14, fontweight='bold')
    
    # 添加图例
    ax.legend(fontsize=11, loc='upper right')
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 保存图像
    if save_path:
        filename = f'{save_path}/3d_traj_sample_{sample_id}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"已保存: {filename}")
    
    # 显示图像
    plt.show()
    
    return fig, ax

def plot_all_trajectories(samples, output_dir='.'):
    """
    绘制所有样本的3D轨迹图
    
    参数:
    samples: 所有样本数据
    output_dir: 输出目录
    """
    print(f"开始绘制{len(samples)}个样本的3D轨迹图...")
    
    for i, sample in enumerate(samples):
        print(f"正在绘制样本 {i}...")
        
        # 绘制单个样本
        plot_3d_trajectory(sample, i, save_path=output_dir)
        
        # 如果样本数量多，可以每绘制一个暂停一下
        if i < len(samples) - 1:
            plt.close('all')  # 关闭当前图形，为下一个做准备
    
    print("所有样本绘制完成！")

def create_summary_plot(samples, save_path=None):
    """
    创建所有样本的汇总图（2x4子图布局）
    
    参数:
    samples: 所有样本数据
    save_path: 保存路径，如果为None则不保存
    """
    # 创建2行4列的子图
    fig = plt.figure(figsize=(20, 10))
    
    for i, sample in enumerate(samples):
        if i >= 8:  # 只绘制前8个样本
            break
            
        # 提取数据
        history = np.array(sample['history'])
        ground_truth = np.array(sample['ground_truth'])
        prediction = np.array(sample['prediction'])
        
        # 添加子图
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        
        # 绘制轨迹
        ax.plot(history[:, 0], history[:, 1], history[:, 2], 
                label='History', color='blue', marker='o', markersize=2)
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], 
                label='Autual', color='green', marker='s', markersize=2)
        ax.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], 
                label='Prediction', color='red', marker='^', markersize=2)
        
        # 设置子图标题
        ax.set_title(f'Sample {i}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
    
    # 添加总标题
    fig.suptitle('3D Trajectory Samples 0-7', fontsize=16, fontweight='bold', y=1.02)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        filename = f'{save_path}/3d_traj_all_samples.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"已保存汇总图: {filename}")
    
    # 显示图像
    plt.show()
    
    return fig

def main():
    """
    主函数：加载数据并绘制轨迹图
    """
    # 数据文件路径
    json_file = '/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/lstm/lstm_result/node_0_trajectories.json'
    
    try:
        # 1. 加载数据
        print("正在加载轨迹数据...")
        samples = load_trajectory_data(json_file)
        
        # 2. 绘制所有样本的3D轨迹图（单独显示）
        print("\n绘制单独样本图...")
        plot_all_trajectories(samples, output_dir='./lstm/lstm_result')
        
        # 3. 创建汇总图（2x4布局）
        print("\n绘制汇总图...")
        create_summary_plot(samples, save_path='./lstm/lstm_result')
        
        print("\n所有绘图任务完成！")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 '{json_file}'")
        print("请确保文件在当前目录下，或提供正确的文件路径。")
    except json.JSONDecodeError:
        print(f"错误：'{json_file}' 文件格式不正确")
    except KeyError as e:
        print(f"错误：JSON文件中缺少必要的键: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    main()
