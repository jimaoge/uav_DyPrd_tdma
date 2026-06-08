import json
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# -------------------------- 1. 配置参数和路径 --------------------------
# json文件实际路径
json_file_path = "/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/results/DQN_training/training_data_history.json"
#json_file_path = "/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/results/DQN_training/training_data_history_edit.json"

# 设置matplotlib使用英文字体
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.unicode_minus': False
})

# 移动平均窗口大小
WINDOW_SIZE = 30  # 可以根据需要调整，建议为5-20

# -------------------------- 2. 定义移动平均函数 --------------------------
def moving_average(data, window_size=WINDOW_SIZE):
    """
    计算移动平均值，保持输出长度与输入相同
    前几个点用较小的窗口
    """
    if window_size <= 1:
        return data
    
    smoothed = []
    for i in range(len(data)):
        if i < window_size:
            # 前window_size-1个点用较小的窗口
            window = i + 1
        else:
            window = window_size
        
        # 计算窗口内数据的平均值
        start_idx = max(0, i - window + 1)
        smoothed.append(np.mean(data[start_idx:i+1]))
    
    return smoothed

# -------------------------- 3. 初始化存储列表 --------------------------
episode_list = []          # X轴: episode number
avg_topo_diff_list = []    # ave_topo_diff (avg per episode)
avg_slot_collision_list = [] # ave_slot_collision (avg per episode)
avg_throughput_list = []   # ave_Throughput (avg per episode)
avg_reward_list = []       # reward (avg per episode)

# -------------------------- 4. 读取并解析JSON数据 --------------------------
if os.path.exists(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        all_episodes_data = json.load(f)
    
    # 遍历每个episode，先求和再求平均
    for episode_data in all_episodes_data:
        ep = episode_data["episode"]
        step_list = episode_data["steps"]
        step_num = len(step_list)  

        sum_topo_diff = 0.0
        sum_slot_collision = 0.0
        sum_throughput = 0.0
        sum_reward = 0.0
        
        # 累加当前episode所有step的指标值
        for step_data in step_list:
            sum_topo_diff += step_data["ave_topo_diff"]
            sum_slot_collision += step_data["ave_slot_collision"]
            sum_throughput += step_data["ave_Throughput"]
            sum_reward += step_data["reward"]
        
        # 计算平均值，防除零报错
        if step_num > 0:
            avg_topo_diff = sum_topo_diff / step_num
            avg_slot_collision = sum_slot_collision / step_num
            avg_throughput = sum_throughput / step_num
            avg_reward = sum_reward / step_num
        else:
            avg_topo_diff = 0.0
            avg_slot_collision = 0.0
            avg_throughput = 0.0
            avg_reward = 0.0
        
        episode_list.append(ep)
        avg_topo_diff_list.append(avg_topo_diff)
        avg_slot_collision_list.append(avg_slot_collision)
        avg_throughput_list.append(avg_throughput)
        avg_reward_list.append(avg_reward)
else:
    print(f"Error: File not found: {json_file_path}")
    exit(1)

# -------------------------- 5. 计算移动平均值 --------------------------
# 对四个维度分别计算移动平均
smoothed_topo_diff = moving_average(avg_topo_diff_list, WINDOW_SIZE)
smoothed_slot_collision = moving_average(avg_slot_collision_list, WINDOW_SIZE)
smoothed_throughput = moving_average(avg_throughput_list, WINDOW_SIZE)
smoothed_reward = moving_average(avg_reward_list, WINDOW_SIZE)

# 保存路径的基础部分
base_save_path = "/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/results/DQN_training/"

# 确保保存目录存在
os.makedirs(base_save_path, exist_ok=True)

# -------------------------- 6. 定义函数：设置刻度 --------------------------
def set_xaxis_ticks(ax, max_episode):
    """
    设置x轴刻度：每50个单位一个主刻度（带数字），每10个单位一个次要刻度
    """
    # 计算需要的主刻度位置（0, 50, 100, 150...）
    major_ticks = np.arange(0, max_episode + 50, 50)
    
    # 设置主刻度位置
    ax.set_xticks(major_ticks)
    
    # 设置主刻度标签格式
    ax.set_xticklabels([f'{int(x)}' for x in major_ticks])
    
    # 设置次要刻度：每10个单位一个
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    
    # 可选：自定义主刻度和次刻度的样式
    ax.tick_params(axis='x', which='major', length=6, width=1.5)
    ax.tick_params(axis='x', which='minor', length=3, width=1)
    
    # 可选：添加网格线
    ax.grid(True, which='major', alpha=0.3, linestyle='-')
    ax.grid(True, which='minor', alpha=0.1, linestyle=':')
    
    return ax

# 计算最大episode数
max_episode = max(episode_list) if episode_list else 0

# -------------------------- 7. 分别绘制4张独立的图（每张图包含原始曲线和移动平均曲线） --------------------------

# 设置统一的颜色和线型
ORIGINAL_COLOR = '#1f77b4'  # 原始曲线颜色
SMOOTHED_COLOR = '#ff7f0e'  # 移动平均曲线颜色
ORIGINAL_LINE_STYLE = '-'  # 实线
SMOOTHED_LINE_STYLE = '-'  # 实线
ORIGINAL_LINE_WIDTH = 1.0
SMOOTHED_LINE_WIDTH = 2.0
ORIGINAL_ALPHA = 0.7
SMOOTHED_ALPHA = 1.0

# 图1: episode vs average topology difference
fig1, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(episode_list, avg_topo_diff_list, 
         color=ORIGINAL_COLOR, 
         linewidth=ORIGINAL_LINE_WIDTH, 
         linestyle=ORIGINAL_LINE_STYLE,
         alpha=ORIGINAL_ALPHA,
         label='Raw Data')
ax1.plot(episode_list, smoothed_topo_diff, 
         color=SMOOTHED_COLOR, 
         linewidth=SMOOTHED_LINE_WIDTH, 
         linestyle=SMOOTHED_LINE_STYLE,
         alpha=SMOOTHED_ALPHA,
         label=f'Moving Average ')
ax1.set_title('Average Topology Difference per Episode', fontsize=16, fontweight='bold')
ax1.set_xlabel('Episode', fontsize=13)
ax1.set_ylabel('Average Topology Difference', fontsize=13)
ax1.legend(fontsize=11, loc='best')

# 设置x轴刻度
set_xaxis_ticks(ax1, max_episode)

plt.tight_layout()
plt.savefig(os.path.join(base_save_path, 'avg_topo_diff_per_episode.png'), dpi=300, bbox_inches='tight')
plt.show()

# 图2: episode vs average slot collision
fig2, ax2 = plt.subplots(figsize=(12, 7))
ax2.plot(episode_list, avg_slot_collision_list, 
         color=ORIGINAL_COLOR, 
         linewidth=ORIGINAL_LINE_WIDTH, 
         linestyle=ORIGINAL_LINE_STYLE,
         alpha=ORIGINAL_ALPHA,
         label='Raw Data')
ax2.plot(episode_list, smoothed_slot_collision, 
         color=SMOOTHED_COLOR, 
         linewidth=SMOOTHED_LINE_WIDTH, 
         linestyle=SMOOTHED_LINE_STYLE,
         alpha=SMOOTHED_ALPHA,
         label=f'Moving Average ')
ax2.set_title('Average Slot Collision per Episode', fontsize=16, fontweight='bold')
ax2.set_xlabel('Episode', fontsize=13)
ax2.set_ylabel('Average Slot Collision', fontsize=13)
ax2.legend(fontsize=11, loc='best')

# 设置x轴刻度
set_xaxis_ticks(ax2, max_episode)

plt.tight_layout()
plt.savefig(os.path.join(base_save_path, 'avg_slot_collision_per_episode.png'), dpi=300, bbox_inches='tight')
plt.show()

# 图3: episode vs average throughput
fig3, ax3 = plt.subplots(figsize=(12, 7))
ax3.plot(episode_list, avg_throughput_list, 
         color=ORIGINAL_COLOR, 
         linewidth=ORIGINAL_LINE_WIDTH, 
         linestyle=ORIGINAL_LINE_STYLE,
         alpha=ORIGINAL_ALPHA,
         label='Raw Data')
ax3.plot(episode_list, smoothed_throughput, 
         color=SMOOTHED_COLOR, 
         linewidth=SMOOTHED_LINE_WIDTH, 
         linestyle=SMOOTHED_LINE_STYLE,
         alpha=SMOOTHED_ALPHA,
         label=f'Moving Average ')
ax3.set_title('Average Throughput per Episode', fontsize=16, fontweight='bold')
ax3.set_xlabel('Episode', fontsize=13)
ax3.set_ylabel('Average Throughput', fontsize=13)
ax3.legend(fontsize=11, loc='best')

# 设置x轴刻度
set_xaxis_ticks(ax3, max_episode)

plt.tight_layout()
plt.savefig(os.path.join(base_save_path, 'avg_throughput_per_episode.png'), dpi=300, bbox_inches='tight')
plt.show()

# 图4: episode vs average reward
fig4, ax4 = plt.subplots(figsize=(12, 7))
ax4.plot(episode_list, avg_reward_list, 
         color=ORIGINAL_COLOR, 
         linewidth=ORIGINAL_LINE_WIDTH, 
         linestyle=ORIGINAL_LINE_STYLE,
         alpha=ORIGINAL_ALPHA,
         label='Raw Data')
ax4.plot(episode_list, smoothed_reward, 
         color=SMOOTHED_COLOR, 
         linewidth=SMOOTHED_LINE_WIDTH, 
         linestyle=SMOOTHED_LINE_STYLE,
         alpha=SMOOTHED_ALPHA,
         label=f'Moving Average ')
ax4.set_title('Average Reward per Episode', fontsize=16, fontweight='bold')
ax4.set_xlabel('Episode', fontsize=13)
ax4.set_ylabel('Average Reward', fontsize=13)
ax4.legend(fontsize=11, loc='best')

# 设置x轴刻度
set_xaxis_ticks(ax4, max_episode)

plt.tight_layout()
plt.savefig(os.path.join(base_save_path, 'avg_reward_per_episode.png'), dpi=300, bbox_inches='tight')
plt.show()
