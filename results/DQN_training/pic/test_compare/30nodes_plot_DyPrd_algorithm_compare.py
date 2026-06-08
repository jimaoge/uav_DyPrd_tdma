import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_throughput_chart(json_path='Dyprd_algorithm_compare.json'):
    """
    绘制吞吐量对比柱状图
    包含3个算法：DyPrd、Prediction、DRAND
    柱状图纯白色，仅用填充图案区分，适配黑白图片
    
    Args:
        json_path: JSON文件路径
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON文件未找到: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取基础数据
    speeds = []
    # 吞吐量数据（3个算法）
    dyprd_throughput = []
    prediction_throughput = []
    drand_throughput = []
    
    for item in data:
        speeds.append(item["speed"])
        
        # 提取吞吐量数据
        dyprd_throughput.append(item["DyPrd"][0]["ave_Throughput"])
        prediction_throughput.append(item["Prediction"][0]["ave_Throughput"])
        drand_throughput.append(item["DRAND"][0]["ave_Throughput"])
    
    # 设置柱状图基础参数（3个算法适配）
    bar_width = 0.25
    x_positions = np.arange(len(speeds))
    
    # 创建画布
    plt.figure(figsize=(12, 6))
    
    # 3个算法：纯白色柱状图，仅用不同填充图案区分
    # DQN-APD Algorithm：左斜线 //
    plt.bar(x_positions - bar_width, dyprd_throughput, width=bar_width, 
            label='DQN-APD Algorithm', color='mediumpurple', edgecolor='black', hatch='//')
    # Prediction Algorithm(Prd=6)：交叉网格 xx
    plt.bar(x_positions, prediction_throughput, width=bar_width, 
            label='Prediction Algorithm(Prd=6)', color='lightgreen', edgecolor='black', hatch='xx')
    # DRAND：十字线 ++
    plt.bar(x_positions + bar_width, drand_throughput, width=bar_width, 
            label='DRAND', color='salmon', edgecolor='black', hatch='++')

        # # DQN-APD Algorithm：左斜线 //
        # plt.bar(x_positions - 1.5*bar_width, dyprd_throughput, width=bar_width, 
        #         label='DQN-APD Algorithm', color='mediumpurple', edgecolor='black', hatch='++')
        # # STTD-TSA(Prd=6)：右斜线 \\
        # plt.bar(x_positions - 0.5*bar_width, tpty_throughput, width=bar_width, 
        #         label='STTD-TSA(Prd=6)', color='skyblue', edgecolor='black', hatch='//')
        # # Prediction Algorithm(Prd=6)：交叉网格 xx
        # plt.bar(x_positions + 0.5*bar_width, prediction_throughput, width=bar_width, 
        #         label='Prediction Algorithm(Prd=6)', color='lightgreen', edgecolor='black', hatch='\\\\')
        # # DRAND：十字线 ++
        # plt.bar(x_positions + 1.5*bar_width, drand_throughput, width=bar_width, 
        #         label='DRAND', color='salmon', edgecolor='black', hatch='xx')
    
    # 设置图表属性
    plt.xlabel('Node Average Speed (m/s)', fontsize=12)
    plt.ylabel('Throughput (Mbps)', fontsize=12)
    plt.xticks(x_positions, speeds)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    
    # 先保存再显示（避免空白图）
    filename = '30nodes_DyPrd_algorithm_throughput_comparison.svg'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"吞吐量对比图已保存为: '{filename}'")
    plt.show()

# 主程序
if __name__ == "__main__":
    try:
        # 绘制吞吐量对比图
        plot_throughput_chart('30nodes_Dyprd_algorithm_compare.json')
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保JSON文件存在于当前目录中。")
    except KeyError as e:
        print(f"错误: JSON数据中缺少关键字 - {e}")
        print("请检查JSON文件结构（确保'DyPrd'、'Prediction'、'DRAND'等关键字存在）。")
    except Exception as e:
        print(f"发生意外错误: {e}")