import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_single_chart(json_path='fixprd_algorithm_compare.json', metric='throughput'):
    """
    绘制图表：吞吐量为单一柱状图，冲突为双轴组合图（柱状图+折线图）
    柱状图纯白色，仅用填充图案区分，适配黑白图片
    
    Args:
        json_path: JSON文件路径
        metric: 指标类型，'throughput'或'collision'
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取基础数据（无论哪种指标都需要吞吐量，冲突率计算依赖）
    speeds = []
    # 吞吐量数据（三种算法）
    tpty_throughput = []
    prediction_throughput = []
    drand_throughput = []
    # 冲突数量数据（三种算法）
    tpty_collision = []
    prediction_collision = []
    drand_collision = []
    
    for item in data:
        speeds.append(item["speed"])
        
        # 提取吞吐量数据（计算冲突率必需）
        tpty_throughput.append(item["TPTY"][0]["ave_Throughput"])
        prediction_throughput.append(item["Prediction"][0]["ave_Throughput"])
        drand_throughput.append(item["DRAND"][0]["ave_Throughput"])
        
        # 提取冲突数量数据（仅冲突图需要）
        if metric == 'collision':
            tpty_collision.append(item["TPTY"][0]["ave_slot_collision"])
            prediction_collision.append(item["Prediction"][0]["ave_slot_collision"])
            drand_collision.append(item["DRAND"][0]["ave_slot_collision"])
    
    # 设置柱状图基础参数
    bar_width = 0.25
    x_positions = np.arange(len(speeds))
    
    # 绘制吞吐量图表（纯白色+填充图案）
    if metric == 'throughput':
        plt.figure(figsize=(10, 6))
        
        # 纯白色柱状图，仅用图案区分
        # STTD-TSA：左斜线 //
        plt.bar(x_positions - bar_width, tpty_throughput, width=bar_width, 
                label='STTD-TSA', color='skyblue', edgecolor='black', hatch='//')
        # Prediction：右斜线 \\
        plt.bar(x_positions, prediction_throughput, width=bar_width, 
                label='Prediction algorithm', color='lightgreen', edgecolor='black', hatch='\\\\')
        # DRAND：网格 xx
        plt.bar(x_positions + bar_width, drand_throughput, width=bar_width, 
                label='DRAND', color='salmon', edgecolor='black', hatch='xx')
        
        # 设置吞吐量图属性
        plt.xlabel('Node Average Speed (m/s)', fontsize=12)
        plt.ylabel('Throughput (Mbps)', fontsize=12)
        plt.xticks(x_positions, speeds)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # 先保存再显示
        filename = f'algorithm_{metric}_comparison.svg'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Graph saved as '{filename}'")
    
    # 绘制冲突双轴组合图（纯白色+填充图案）
    else:
        # 创建双轴画布
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 左轴：冲突数量（纯白色柱状图+填充图案）
        ax1.bar(x_positions - bar_width, tpty_collision, width=bar_width, 
                label='STTD-TSA (Collision)', color='skyblue', edgecolor='black', hatch='//')
        ax1.bar(x_positions, prediction_collision, width=bar_width, 
                label='Prediction algorithm (Collision)', color='lightgreen', edgecolor='black', hatch='\\\\')
        ax1.bar(x_positions + bar_width, drand_collision, width=bar_width, 
                label='DRAND (Collision)', color='salmon', edgecolor='black', hatch='xx')
        
        # 设置左轴属性
        ax1.set_xlabel('Node Average Speed (m/s)', fontsize=12)
        ax1.set_ylabel('Slot Collision', fontsize=12)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(speeds)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 计算冲突率：冲突数量/(冲突数量+吞吐量)
        tpty_rate = np.array(tpty_collision) / (np.array(tpty_collision) + np.array(tpty_throughput))
        prediction_rate = np.array(prediction_collision) / (np.array(prediction_collision) + np.array(prediction_throughput))
        drand_rate = np.array(drand_collision) / (np.array(drand_collision) + np.array(drand_throughput))
        
        # 右轴：冲突率（折线图）
        ax2 = ax1.twinx()
        ax2.plot(x_positions, tpty_rate, marker='o', linestyle='-', color='darkblue', 
                 label='STTD-TSA (Collision Rate)')
        ax2.plot(x_positions, prediction_rate, marker='s', linestyle='-', color='darkgreen', 
                 label='Prediction algorithm (Collision Rate)')
        ax2.plot(x_positions, drand_rate, marker='^', linestyle='-', color='darkred', 
                 label='DRAND (Collision Rate)')
        
        # 设置右轴属性
        ax2.set_ylabel('Collision Rate', fontsize=12)
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 设置标题和布局
        plt.tight_layout()
        
        # 先保存再显示
        filename = f'algorithm_{metric}_comparison.svg'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Graph saved as '{filename}'")

# 主程序
if __name__ == "__main__":
    try:
        plot_single_chart('fixprd_algorithm_compare.json', 'throughput')
        plot_single_chart('fixprd_algorithm_compare.json', 'collision')
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'fixprd_algorithm_compare.json' exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {e}")