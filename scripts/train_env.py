import sys
import json
import numpy as np
import os
from decimal import Decimal
import torch
from pathlib import Path
# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from environment.env import SelfOrganizingNetworkEnv
from DRL.DQN_model import DQN
from config.config import config
from utils.utils import NumpyEncoder

def run():
    env = SelfOrganizingNetworkEnv()
    agent = DQN()
    epsilon_start = config.DQN_PARAMS['epsilon_start']
    epsilon_end = config.DQN_PARAMS['epsilon_end']
    epsilon_decay = config.DQN_PARAMS['epsilon_decay']
    Epsilon = epsilon_start
    
    data_dir = config.DATA_PATHS['DQN_train_result_path']   # "results" / "DQN_training"
    os.makedirs(data_dir, exist_ok=True)  # 创建保存数据的目录
    
    all_episodes_data = []  # 存储所有episode的数据
    env.load_scalers_list() # 加载归一化器
    print("所有节点归一化器已加载")
    total_episodes = config.DQN_PARAMS['total_episode_in_train']
    for ep in range(total_episodes):
        print()
        # Epsilon衰减
        Epsilon = max(epsilon_end, epsilon_decay * Epsilon)
        state = env.reset_random_time()
        
        # 记录每个episode的数据
        episode_data = {
            "episode": ep,
            "steps": []
        }        
        total_step_count = config.DQN_PARAMS['max_steps_per_episode']
        for step_count in range(total_step_count):
            print()
            print(f"-----当前step_count:{step_count}/{total_step_count},episode:{ep}/{total_episodes}-----")
            state = env.get_current_state()
            action = agent.choose_action(state, Epsilon)
            state_next, reward, done = env.step(action)
            agent.remember(state, action, state_next, reward)
            agent.train()  # 在线学习

            state_list = state.tolist()  # 转列表
            state_15 = state_list[:-1]   # 去掉最后一位，保留前15位
            # 手动切分成5组，每组3个值
            state_5x3 = [state_15[i*3 : (i+1)*3] for i in range(5)]
            # 按列求平均值
            link_dynamic = [
                sum(col) / len(col) for col in zip(*state_5x3)
            ]            
            # 记录当前step的数据
            step_data = {
                "step": step_count,
                "link_dynamic": link_dynamic,
                "action": int(action),
                "DyPrd": env.DyPrd,
                "topo_diff": sum(env.diff),
                "slot_collision": env.coll,
                "Throughput": env.T_sum,
                "reward": reward
            }
            
            episode_data["steps"].append(step_data)
            
        # 保存当前episode的数据到内存
        all_episodes_data.append(episode_data)      
        
        # 只在定期保存检查点时保存数据
        save_interval = 50
        if (ep + 1) % save_interval == 0:
            # 保存模型检查点
            agent.save_model(filename=f"dqn_checkpoint_ep{ep}.pth")
            
            # # 只保存当前检查点周期的数据（最后save_interval个episode）
            # start_ep = max(0, ep - save_interval + 1)
            # recent_episodes = all_episodes_data[start_ep:ep+1]
            
            # # 保存数据检查点
            # data_path = os.path.join(data_dir, f"training_data_ep{start_ep}_to_ep{ep}.json")
            # with open(data_path, 'w', encoding='utf-8') as f:
            #     json.dump(recent_episodes, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    # 训练结束后保存最终模型
    agent.save_model()
    
    # 保存最终完整数据
    final_data_path = os.path.join(data_dir, "training_data_final.json")
    with open(final_data_path, 'w', encoding='utf-8') as f:
        json.dump(all_episodes_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print("训练完成")
            
            
if __name__ == "__main__":
    run()          
    
    