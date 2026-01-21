import json
import matplotlib.pyplot as plt
import os

# -------------------------- 1. 配置参数和路径 --------------------------
# json文件实际路径
json_file_path = "/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/results/DQN_training/60-120training_data_history.json"

plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 2. 初始化存储列表 --------------------------
episode_list = []          # X轴: episode number
avg_topo_diff_list = []    # ave_topo_diff (avg per episode)
avg_slot_collision_list = [] # ave_slot_collision (avg per episode)
avg_throughput_list = []   # ave_Throughput (avg per episode)
avg_reward_list = []       # reward (avg per episode)

# -------------------------- 3. 读取并解析JSON数据 --------------------------
if os.path.exists(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        all_episodes_data = json.load(f)
    
    # 遍历每个episode，先求和再求平均（严格按你的要求）
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

# -------------------------- 4. 绘制4张子图（2行2列） --------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
# 全部改为英文标题，无字体问题
fig.suptitle('Training Metrics - Average per Episode', fontsize=16, fontweight='bold')

# 子图1: episode vs average topology difference
axes[0,0].plot(episode_list, avg_topo_diff_list, color='#1f77b4', linewidth=1.5, marker='.', markersize=3)
axes[0,0].set_title('Average Topology Difference', fontsize=12)
axes[0,0].set_xlabel('Episode')
axes[0,0].set_ylabel('ave_topo_diff (Avg)')
axes[0,0].grid(alpha=0.3, linestyle='-')

# 子图2: episode vs average slot collision
axes[0,1].plot(episode_list, avg_slot_collision_list, color='#ff7f0e', linewidth=1.5, marker='.', markersize=3)
axes[0,1].set_title('Average Slot Collision', fontsize=12)
axes[0,1].set_xlabel('Episode')
axes[0,1].set_ylabel('ave_slot_collision (Avg)')
axes[0,1].grid(alpha=0.3, linestyle='-')

# 子图3: episode vs average throughput
axes[1,0].plot(episode_list, avg_throughput_list, color='#2ca02c', linewidth=1.5, marker='.', markersize=3)
axes[1,0].set_title('Average Throughput', fontsize=12)
axes[1,0].set_xlabel('Episode')
axes[1,0].set_ylabel('ave_Throughput (Avg)')
axes[1,0].grid(alpha=0.3, linestyle='-')

# 子图4: episode vs average reward
axes[1,1].plot(episode_list, avg_reward_list, color='#d62728', linewidth=1.5, marker='.', markersize=3)
axes[1,1].set_title('Average Reward', fontsize=12)
axes[1,1].set_xlabel('Episode')
axes[1,1].set_ylabel('Reward (Avg)')
axes[1,1].grid(alpha=0.3, linestyle='-')

# 自动调整间距
plt.tight_layout(rect=[0, 0, 1, 0.96])

# -------------------------- 5. 显示+保存图片 --------------------------
plt.show()
# 保存高清图片
fig.savefig('/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/results/DQN_training/training_metrics_episode_avg.png', dpi=300, bbox_inches='tight')