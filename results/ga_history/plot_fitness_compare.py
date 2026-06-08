import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取两个JSON文件
def load_fitness_data():
    slot_path = '/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/results/ga_history/fitness_data.json'
    compare_path = '/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/results/ga_history/compare_fitness_data.json'
    
    with open(slot_path, 'r') as f:
        slot_data = json.load(f)
    
    with open(compare_path, 'r') as f:
        node_data = json.load(f)
    
    return slot_data, node_data, slot_path, compare_path

# 绘制适应度对比曲线
def plot_fitness_compare():
    # 加载数据
    slot_data, node_data, slot_path, compare_path = load_fitness_data()
    
    # 提取适应度历史数据
    slot_fitness = slot_data['max_fitness_history']
    node_fitness = node_data['max_fitness_history']
    
    # 创建横坐标（generation）
    slot_generations = list(range(len(slot_fitness)))
    node_generations = list(range(len(node_fitness)))
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制两条曲线
    plt.plot(slot_generations, slot_fitness, linewidth=2, label='SBCC-GA')
    plt.plot(node_generations, node_fitness, linewidth=2, linestyle='--', label='GA')
    
    # 新增代码：添加Greedy Algorithm水平横线
    plt.axhline(y=-94, linewidth=2, linestyle='-.', label='Greedy Algorithm')
    
    # 设置图形属性
    plt.xlabel('Generation', fontsize=14)
    plt.ylabel('Fitness', fontsize=14)
    # ===================== 修改处：图例放置在右下角 =====================
    plt.legend(fontsize=12, loc='lower right')
    # ==================================================================
    plt.grid(True, alpha=0.3)
    
    # 设置坐标轴范围
    max_generations = max(len(slot_fitness), len(node_fitness))
    plt.xlim(0, max_generations - 1)
    
    # 保存图片
    save_dir = os.path.dirname(slot_path)
    save_path = os.path.join(save_dir, 'fitness_compare.svg')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {save_path}")
    
    # 显示图形
    plt.show()

# 主程序
if __name__ == "__main__":
    plot_fitness_compare()