import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_single_chart(json_path='fixprd_algorithm_compare.json', metric='throughput'):
    """
    绘制单个指标的柱状图
    
    Args:
        json_path: JSON文件路径
        metric: 指标类型，'throughput'或'collision'
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取数据
    speeds = []
    tpty_values = []
    prediction_values = []
    drand_values = []
    
    for item in data:
        speeds.append(item["speed"])
        
        if metric == 'throughput':
            tpty_values.append(item["TPTY"][0]["ave_Throughput"])
            prediction_values.append(item["Prediction"][0]["ave_Throughput"])
            drand_values.append(item["DRAND"][0]["ave_Throughput"])
        else:  # collision
            tpty_values.append(item["TPTY"][0]["ave_slot_collision"])
            prediction_values.append(item["Prediction"][0]["ave_slot_collision"])
            drand_values.append(item["DRAND"][0]["ave_slot_collision"])
    
    # 设置柱状图参数
    bar_width = 0.25
    x_positions = np.arange(len(speeds))
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 绘制柱状图
    plt.bar(x_positions - bar_width, tpty_values, width=bar_width, label='Proposed Algorithm', color='skyblue', edgecolor='black')
    plt.bar(x_positions, prediction_values, width=bar_width, label='Prediction', color='lightgreen', edgecolor='black')
    plt.bar(x_positions + bar_width, drand_values, width=bar_width, label='DRAND', color='salmon', edgecolor='black')
    
    # 设置图形属性
    plt.xlabel('Node Average Speed (m/s)', fontsize=12)
    
    if metric == 'throughput':
        plt.ylabel('Throughput (Mbps)', fontsize=12)
        plt.title('Throughput Comparison of Different Algorithms (Prd = 6)', fontsize=14, fontweight='bold')
    else:
        plt.ylabel('Slot Collision', fontsize=12)
        plt.title('Slot Collision Comparison of Different Algorithms (Prd = 6)', fontsize=14, fontweight='bold')
    
    plt.xticks(x_positions, speeds)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # # 在柱状图上添加数值标签
    # for i, (tpty, pred, drand) in enumerate(zip(tpty_values, prediction_values, drand_values)):
    #     plt.text(i - bar_width, tpty + 0.1, f'{tpty:.2f}', ha='center', va='bottom', fontsize=9)
    #     plt.text(i, pred + 0.1, f'{pred:.2f}', ha='center', va='bottom', fontsize=9)
    #     plt.text(i + bar_width, drand + 0.1, f'{drand:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()
    
    # 保存图形
    filename = f'algorithm_{metric}_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
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