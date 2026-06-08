import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_single_chart(json_path='Dyprd_algorithm_compare.json', metric='throughput'):
    """
    绘制图表：吞吐量为单一柱状图，冲突为双轴组合图（柱状图+折线图）
    包含4个算法：DyPrd、Fixed_Prd(Prd=6)、Prediction(Prd=6)、DRAND
    柱状图纯白色，仅用填充图案区分，适配黑白图片
    
    Args:
        json_path: JSON文件路径
        metric: 指标类型，'throughput'（吞吐量）或'collision'（时隙冲突）
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON文件未找到: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取基础数据（冲突率计算依赖吞吐量，无论哪种指标都需提取）
    speeds = []
    # 吞吐量数据（4个算法）
    dyprd_throughput = []
    tpty_throughput = []
    prediction_throughput = []
    drand_throughput = []
    # 冲突数量数据（4个算法，仅冲突图需要）
    dyprd_collision = []
    tpty_collision = []
    prediction_collision = []
    drand_collision = []
    
    for item in data:
        speeds.append(item["speed"])
        
        # 提取吞吐量数据（计算冲突率必需）
        dyprd_throughput.append(item["DyPrd"][0]["ave_Throughput"])
        tpty_throughput.append(item["TPTY"][0]["ave_Throughput"])
        prediction_throughput.append(item["Prediction"][0]["ave_Throughput"])
        drand_throughput.append(item["DRAND"][0]["ave_Throughput"])
        
        # 提取冲突数量数据（仅冲突图需要）
        if metric == 'collision':
            dyprd_collision.append(item["DyPrd"][0]["ave_slot_collision"])
            tpty_collision.append(item["TPTY"][0]["ave_slot_collision"])
            prediction_collision.append(item["Prediction"][0]["ave_slot_collision"])
            drand_collision.append(item["DRAND"][0]["ave_slot_collision"])
    
    # 设置柱状图基础参数（4个算法适配）
    bar_width = 0.2
    x_positions = np.arange(len(speeds))
    
    # 绘制吞吐量图表（纯白色+填充图案区分）
    if metric == 'throughput':
        plt.figure(figsize=(12, 6))
        
        # 4个算法：纯白色柱状图，仅用不同填充图案区分
        # DQN-APD Algorithm：左斜线 //
        plt.bar(x_positions - 1.5*bar_width, dyprd_throughput, width=bar_width, 
                label='DQN-APD Algorithm', color='mediumpurple', edgecolor='black', hatch='++')
        # STTD-TSA(Prd=6)：右斜线 \\
        plt.bar(x_positions - 0.5*bar_width, tpty_throughput, width=bar_width, 
                label='STTD-TSA(Prd=6)', color='skyblue', edgecolor='black', hatch='//')
        # Prediction Algorithm(Prd=6)：交叉网格 xx
        plt.bar(x_positions + 0.5*bar_width, prediction_throughput, width=bar_width, 
                label='Prediction Algorithm(Prd=6)', color='lightgreen', edgecolor='black', hatch='\\\\')
        # DRAND：十字线 ++
        plt.bar(x_positions + 1.5*bar_width, drand_throughput, width=bar_width, 
                label='DRAND', color='salmon', edgecolor='black', hatch='xx')
        
        # 设置吞吐量图属性
        plt.xlabel('Node Average Speed (m/s)', fontsize=12)
        plt.ylabel('Throughput (Mbps)', fontsize=12)
        plt.xticks(x_positions, speeds)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3, linestyle='--', axis='y')
        plt.tight_layout()
        
        # 先保存再显示（避免空白图）
        filename = f'DyPrd_algorithm_{metric}_comparison.svg'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: '{filename}'")
        plt.show()
    
    # 绘制冲突双轴组合图（纯白色+填充图案区分）
    else:
        # 创建双轴画布
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 左轴：4个算法的冲突数量（纯白色柱状图+专属图案）
        ax1.bar(x_positions - 1.5*bar_width, dyprd_collision, width=bar_width, 
                label='DQN-APD Algorithm (Collision)', color='mediumpurple', edgecolor='black', hatch='++')
        ax1.bar(x_positions - 0.5*bar_width, tpty_collision, width=bar_width, 
                label='STTD-TSA(Prd=6) (Collision)', color='skyblue', edgecolor='black', hatch='//')
        ax1.bar(x_positions + 0.5*bar_width, prediction_collision, width=bar_width, 
                label='Prediction Algorithm(Prd=6) (Collision)', color='lightgreen', edgecolor='black', hatch='\\\\')
        ax1.bar(x_positions + 1.5*bar_width, drand_collision, width=bar_width, 
                label='DRAND (Collision)', color='salmon', edgecolor='black', hatch='xx')
        
        # 设置左轴属性
        ax1.set_xlabel('Node Average Speed (m/s)', fontsize=12)
        ax1.set_ylabel('Slot Collision', fontsize=12)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(speeds)
        ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 计算4个算法的冲突率：冲突数量/(冲突数量+吞吐量)
        dyprd_rate = np.array(dyprd_collision) / (np.array(dyprd_collision) + np.array(dyprd_throughput))
        tpty_rate = np.array(tpty_collision) / (np.array(tpty_collision) + np.array(tpty_throughput))
        prediction_rate = np.array(prediction_collision) / (np.array(prediction_collision) + np.array(prediction_throughput))
        drand_rate = np.array(drand_collision) / (np.array(drand_collision) + np.array(drand_throughput))
        
        # 右轴：4个算法的冲突率（折线图，不同标记区分）
        ax2 = ax1.twinx()
        ax2.plot(x_positions, dyprd_rate, marker='o', linestyle='-', color='darkviolet', 
                 label='DQN-APD Algorithm (Collision Rate)', markersize=6)
        ax2.plot(x_positions, tpty_rate, marker='s', linestyle='-', color='darkblue', 
                 label='STTD-TSA(Prd=6) (Collision Rate)', markersize=6)
        ax2.plot(x_positions, prediction_rate, marker='^', linestyle='-', color='darkgreen', 
                 label='Prediction Algorithm(Prd=6) (Collision Rate)', markersize=6)
        ax2.plot(x_positions, drand_rate, marker='*', linestyle='-', color='darkred', 
                 label='DRAND (Collision Rate)', markersize=6)
        
        # 设置右轴属性
        ax2.set_ylabel('Collision Rate', fontsize=12)
        
        # 合并双轴图例并放置在左上角
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
        
        # 设置标题和布局
        plt.tight_layout()
        
        # 先保存再显示
        filename = f'DyPrd_algorithm_{metric}_comparison.svg'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图表已保存为: '{filename}'")
        plt.show()

# 主程序
if __name__ == "__main__":
    try:
        # 绘制吞吐量对比图
        plot_single_chart('Dyprd_algorithm_compare.json', 'throughput')
        # 绘制时隙冲突双轴组合图
        plot_single_chart('Dyprd_algorithm_compare.json', 'collision')
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保JSON文件存在于当前目录中。")
    except KeyError as e:
        print(f"错误: JSON数据中缺少关键字 - {e}")
        print("请检查JSON文件结构（确保'DyPrd'等关键字存在）。")
    except Exception as e:
        print(f"发生意外错误: {e}")