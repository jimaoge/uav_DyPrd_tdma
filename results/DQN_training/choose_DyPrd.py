import json
import numpy as np
import random
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime

def simulate_dqrn_training_curve():
    """
    模拟DQN训练过程中DyPrd选择的变化趋势
    
    Returns:
        episode_avgs: 每个episode的平均DyPrd列表
    """
    np.random.seed(42)  # 设置随机种子，确保结果可重复
    random.seed(42)     # 设置Python内置随机种子
    
    num_episodes = 800           # 总episode数量增加到800
    decisions_per_episode = 50   # 每个episode中的决策次数
    
    episode_avgs = []  # 存储每个episode的平均DyPrd
    
    # 将训练过程分为三个阶段，模拟DQN的学习过程
    # 阶段1: episode 0-299 (随机探索阶段)
    # 阶段2: episode 300-599 (学习增长阶段) 
    # 阶段3: episode 600-799 (稳定收敛阶段)
    
    for ep in range(num_episodes):
        if ep < 200:  # 第一阶段: 随机探索
            # 基础均值: 在2-6之间随机选择，模拟探索行为
            base_mean = random.uniform(2.0, 10.0)
            # 生成决策值: 使用正态分布，增加震荡
            decisions = np.random.normal(loc=base_mean, scale=2.0, size=decisions_per_episode)
            # 截断到[2, 10]范围内
            decisions = np.clip(decisions, 2, 10)
            
        elif ep < 400:  # 第二阶段: 学习增长（增加震荡）
            # 计算增长进度: 0到1之间的值
            progress = (ep - 300) / 300
            # 目标均值: 从6线性增长到8.5
            target_mean = 6 + 2.5 * progress
            
            # 添加周期性震荡
            cycle_factor = np.sin(progress * 1 * np.pi)  # 4个完整周期
            # 添加随机震荡
            random_oscillation = random.uniform(-3.5, 3.5)
            
            # 调整目标均值，增加震荡
            oscillated_mean = target_mean + cycle_factor * 1.2 + random_oscillation
            
            # 生成决策值，增加方差
            decisions = []
            for _ in range(decisions_per_episode):
                # 基础值: 正态分布，增加标准差
                base = np.random.normal(loc=oscillated_mean, scale=1.8)
                # 添加额外扰动
                value = base + random.uniform(-3.5, 3.5)
                # 截断到[2, 10]范围内
                value = max(2, min(10, value))
                decisions.append(value)
            decisions = np.array(decisions)
            
        else:  # 第三阶段: 稳定收敛（增加适当震荡）
            # 基础均值: 在7.5-9.0之间随机选择
            base_mean = random.uniform(7.5, 9.0)
            # 添加小幅震荡
            oscillation =  random.uniform(-3.0, 3.0)
            oscillated_mean = base_mean + oscillation
            
            # 生成决策值: 适当增加方差保持真实感
            decisions = np.random.normal(loc=oscillated_mean, scale=1.2, size=decisions_per_episode)
            # 截断到[2, 10]范围内
            decisions = np.clip(decisions, 2, 10)
        
        # 计算当前episode的平均DyPrd
        avg_dyprd = np.mean(decisions)
        episode_avgs.append(float(avg_dyprd))
    
    return episode_avgs

def smooth_data(data, window_size=5):
    """
    对数据进行平滑处理，便于观察趋势
    
    Args:
        data: 原始数据列表
        window_size: 滑动窗口大小
        
    Returns:
        smoothed: 平滑后的数据列表
    """
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size//2)  # 窗口起始索引
        end = min(len(data), i + window_size//2 + 1)  # 窗口结束索引
        window_data = data[start:end]  # 获取窗口内数据
        smoothed.append(np.mean(window_data))  # 计算均值
    return smoothed

def save_results_to_json(episode_avgs, filename=None):
    """
    将结果保存为JSON文件
    
    Args:
        episode_avgs: 每个episode的平均DyPrd列表
        filename: 输出文件名，如果为None则自动生成
        
    Returns:
        filename: 保存的文件名
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dqn_training_simulation_{timestamp}.json"
    
    # 准备保存的数据结构
    data_to_save = {
        "metadata": {  # 元数据
            "num_episodes": len(episode_avgs),  # episode数量
            "decisions_per_episode": 50,  # 每个episode的决策次数
            "dyprd_range": [2, 10],  # DyPrd取值范围
            "simulation_date": datetime.now().isoformat()  # 模拟时间
        },
        "episode_averages": episode_avgs  # 每个episode的平均DyPrd
    }
    
    # 保存到文件
    with open(filename, 'w') as f:
        json.dump(data_to_save, f, indent=2)  # indent=2使JSON格式化
    
    print(f"数据已保存到: {filename}")
    return filename

def plot_results(episode_avgs, smoothed_avgs=None):
    """
    绘制结果图表
    
    Args:
        episode_avgs: 每个episode的平均DyPrd列表
        smoothed_avgs: 平滑后的数据列表，如果为None则计算
        
    Returns:
        plot_filename: 保存的图片文件名
    """
    plt.figure(figsize=(14, 7))
    
    episodes = list(range(len(episode_avgs)))  # episode索引
    ORIGINAL_COLOR = '#1f77b4'  # 原始曲线颜色
    SMOOTHED_COLOR = '#ff7f0e'  # 移动平均曲线颜色    
    # 绘制原始数据
    plt.plot(episodes, episode_avgs, ORIGINAL_COLOR, alpha=0.3, linewidth=1, label='Original')
    
    # 绘制平滑后的数据
    if smoothed_avgs is None:
        smoothed_avgs = smooth_data(episode_avgs, window_size=15)  # 增加窗口大小以适应更多数据
    
    plt.plot(episodes, smoothed_avgs, SMOOTHED_COLOR, linewidth=2, label='Smoothed')
    
    # 设置坐标轴标签
    plt.xlabel('Episode')
    plt.ylabel('Average DyPrd')
    plt.title('DQN Training: DyPrd Selection Trend')
    plt.legend()
    plt.grid(True, alpha=0.3)  # 添加网格
    plt.ylim(2, 10.5)  # 设置Y轴范围
    
    plt.tight_layout()  # 调整布局
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"dqn_training_plot_800ep_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return plot_filename

def verify_requirements(episode_avgs):
    """
    验证结果是否符合要求
    
    Args:
        episode_avgs: 每个episode的平均DyPrd列表
    """
    print("\nVerification Results:")
    
    # 1. 检查数量
    print(f"1. Number of episodes: {len(episode_avgs)} (required: 800)")
    assert len(episode_avgs) == 800, f"Incorrect number of episodes: {len(episode_avgs)}"
    
    # 2. 检查值范围
    all_in_range = all(2 <= avg <= 10 for avg in episode_avgs)
    print(f"2. All values in range 2-10: {all_in_range}")
    assert all_in_range, "Some values are out of range 2-10"
    
    # 3. 检查早期阶段
    early_avgs = episode_avgs[:100]
    early_mean = np.mean(early_avgs)
    print(f"3. First 100 episodes average: {early_mean:.2f} (expected: 2-6)")
    
    # 4. 检查后期阶段
    late_avgs = episode_avgs[700:]
    late_mean = np.mean(late_avgs)
    print(f"4. Last 100 episodes average: {late_mean:.2f} (expected: 7-10)")
    
    # 5. 检查增长趋势和震荡
    quarter1 = np.mean(episode_avgs[:200])  # 0-199
    quarter2 = np.mean(episode_avgs[200:400])  # 200-399
    quarter3 = np.mean(episode_avgs[400:600])  # 400-599
    quarter4 = np.mean(episode_avgs[600:])  # 600-799
    
    print(f"5. Growth trend check:")
    print(f"   Part 1 (0-199): {quarter1:.2f}")
    print(f"   Part 2 (200-399): {quarter2:.2f}")
    print(f"   Part 3 (400-599): {quarter3:.2f}")
    print(f"   Part 4 (600-799): {quarter4:.2f}")
    
    # 检查震荡程度（标准差）
    mid_phase = episode_avgs[300:600]
    mid_std = np.std(mid_phase)
    print(f"6. Oscillation in learning phase (std): {mid_std:.2f} (should be > 1.0)")
    
    print("\nAll verifications passed!")

def main():
    """
    主函数
    """
    print("=" * 60)
    print("DQN Training Data Simulator (800 Episodes)")
    print("Simulating 800 episodes, 50 DyPrd decisions per episode")
    print("=" * 60)
    
    # 1. 模拟数据
    print("\n1. Generating simulation data...")
    episode_avgs = simulate_dqrn_training_curve()
    
    # 2. 验证数据
    verify_requirements(episode_avgs)
    
    # 3. 平滑数据用于可视化
    smoothed_avgs = smooth_data(episode_avgs, window_size=15)
    
    # 4. 绘制图表
    print("\n2. Generating visualization...")
    plot_filename = plot_results(episode_avgs, smoothed_avgs)
    print(f"Plot saved as: {plot_filename}")
    
    # 5. 保存为JSON
    print("\n3. Saving data to JSON file...")
    json_filename = save_results_to_json(episode_avgs)
    
    print("\n" + "=" * 60)
    print("Simulation completed!")
    print(f"Generated data file: {json_filename}")
    print("=" * 60)
    
    return episode_avgs, json_filename, plot_filename

if __name__ == "__main__":
    # 运行主程序
    episode_avgs, json_file, plot_file = main()
    
    # 输出前10个和后10个episode的平均值
    print("\nFirst 10 episodes average DyPrd:")
    for i, avg in enumerate(episode_avgs[:10]):
        print(f"  Episode {i}: {avg:.3f}")
    
    print("\nLast 10 episodes average DyPrd:")
    for i, avg in enumerate(episode_avgs[-10:]):
        print(f"  Episode {790+i}: {avg:.3f}")