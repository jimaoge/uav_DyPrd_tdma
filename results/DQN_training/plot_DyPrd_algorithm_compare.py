import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_single_chart(json_path='fixprd_algorithm_compare.json', metric='throughput'):
    """
    绘制单个指标的柱状图（包含DyPrd算法）
    
    Args:
        json_path: JSON文件路径
        metric: 指标类型，'throughput'（吞吐量）或'collision'（时隙冲突）
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取数据
    speeds = []
    dyprd_values = []  # DyPrd算法指标值
    tpty_values = []    # TPTY算法指标值
    prediction_values = []  # Prediction算法指标值
    drand_values = []   # DRAND算法指标值
    
    for item in data:
        speeds.append(item["speed"])
        
        if metric == 'throughput':
            dyprd_values.append(item["DyPrd"][0]["ave_Throughput"])
            tpty_values.append(item["TPTY"][0]["ave_Throughput"])
            prediction_values.append(item["Prediction"][0]["ave_Throughput"])
            drand_values.append(item["DRAND"][0]["ave_Throughput"])
        else:  # collision
            dyprd_values.append(item["DyPrd"][0]["ave_slot_collision"])
            tpty_values.append(item["TPTY"][0]["ave_slot_collision"])
            prediction_values.append(item["Prediction"][0]["ave_slot_collision"])
            drand_values.append(item["DRAND"][0]["ave_slot_collision"])
    
    # 设置柱状图参数（4个算法，调整宽度和偏移）
    bar_width = 0.2  # 减小宽度以容纳4个柱子
    x_positions = np.arange(len(speeds))
    
    # 创建图形
    plt.figure(figsize=(12, 6))  # 加宽画布，避免柱子拥挤
    
    # 绘制4个算法的柱状图（调整偏移位置）
    plt.bar(x_positions - 1.5*bar_width, dyprd_values, width=bar_width, 
            label='DyPrd Algorithm', color='mediumpurple', edgecolor='black')
    plt.bar(x_positions - 0.5*bar_width, tpty_values, width=bar_width, 
            label='Fixed_Prd(Prd=6) Algorithm', color='skyblue', edgecolor='black')
    plt.bar(x_positions + 0.5*bar_width, prediction_values, width=bar_width, 
            label='Prediction(Prd=6) Algorithm', color='lightgreen', edgecolor='black')
    plt.bar(x_positions + 1.5*bar_width, drand_values, width=bar_width, 
            label='DRAND', color='salmon', edgecolor='black')
    
    # 设置图形属性
    plt.xlabel('Node Average Speed (m/s)', fontsize=12)
    
    if metric == 'throughput':
        plt.ylabel('Throughput (Mbps)', fontsize=12)
        plt.title('Throughput Comparison of Different Algorithms', 
                  fontsize=14, fontweight='bold')
    else:
        plt.ylabel('Slot Collision', fontsize=12)
        plt.title('Slot Collision Comparison of Different Algorithms', 
                  fontsize=14, fontweight='bold')
    
    plt.xticks(x_positions, speeds)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')  # 仅显示y轴网格，更清晰
    
    # # 在柱状图上添加数值标签（按需开启/关闭）
    # for i, (dyprd, tpty, pred, drand) in enumerate(zip(dyprd_values, tpty_values, prediction_values, drand_values)):
    #     # 控制标签显示位置（避免超出画布）
    #     offset = 0.1 if metric == 'throughput' else 0.05
    #     plt.text(i - 1.5*bar_width, dyprd + offset, f'{dyprd:.2f}', 
    #              ha='center', va='bottom', fontsize=9)
    #     plt.text(i - 0.5*bar_width, tpty + offset, f'{tpty:.2f}', 
    #              ha='center', va='bottom', fontsize=9)
    #     plt.text(i + 0.5*bar_width, pred + offset, f'{pred:.2f}', 
    #              ha='center', va='bottom', fontsize=9)
    #     plt.text(i + 1.5*bar_width, drand + offset, f'{drand:.2f}', 
    #              ha='center', va='bottom', fontsize=9)
    
    # 调整布局
    plt.tight_layout()
    
    # 先保存再显示（修复plt.show()后保存空白的问题）
    filename = f'DyPrd_algorithm_{metric}_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph saved as '{filename}'")
    
    # 显示图形
    plt.show()

# 主程序
if __name__ == "__main__":
    try:
        # 绘制吞吐量对比图
        plot_single_chart('Dyprd_algorithm_compare.json', 'throughput')
        # 绘制时隙冲突对比图
        plot_single_chart('Dyprd_algorithm_compare.json', 'collision')
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the JSON file exists in the current directory.")
    except KeyError as e:
        print(f"Error: Missing key in JSON data - {e}")
        print("Please check the JSON file structure (ensure 'DyPrd' key exists).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")