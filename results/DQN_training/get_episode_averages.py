import json
import statistics
from typing import List, Dict, Any

def process_dqn_records(input_file: str, output_file: str) -> None:
    """
    处理DQN训练记录，计算每个episode的平均值
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    try:
        # 1. 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 处理每个episode
        processed_data = []
        
        for episode_data in data:
            episode_num = episode_data["episode"]
            steps = episode_data["steps"]
            
            # 提取每个指标的所有值
            topo_diffs = []
            slot_collisions = []
            throughputs = []
            rewards = []
            
            for step in steps:
                topo_diffs.append(step["ave_topo_diff"])
                slot_collisions.append(step["ave_slot_collision"])
                throughputs.append(step["ave_Throughput"])
                rewards.append(step["reward"])
            
            # 计算平均值
            avg_episode = {
                "episode": episode_num,
                "avg_ave_topo_diff": statistics.mean(topo_diffs),
                "avg_ave_slot_collision": statistics.mean(slot_collisions),
                "avg_ave_Throughput": statistics.mean(throughputs),
                "avg_reward": statistics.mean(rewards)
            }
            
            processed_data.append(avg_episode)
        
        # 3. 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"处理完成！共处理了 {len(processed_data)} 个episode")
        print(f"结果已保存到: {output_file}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except KeyError as e:
        print(f"错误：JSON格式不正确，缺少必要的键: {e}")
    except json.JSONDecodeError:
        print(f"错误：文件 {input_file} 不是有效的JSON格式")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

def process_no_reward_records(input_file: str, output_file: str) -> None:
    """
    处理DQN训练记录，计算每个episode的平均值
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    try:
        # 1. 读取原始JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 处理每个episode
        processed_data = []
        
        for episode_data in data:
            episode_num = episode_data["episode"]
            steps = episode_data["steps"]
            
            # 提取每个指标的所有值
            topo_diffs = []
            slot_collisions = []
            throughputs = []
            
            
            for step in steps:
                topo_diffs.append(step["ave_topo_diff"])
                slot_collisions.append(step["ave_slot_collision"])
                throughputs.append(step["ave_Throughput"])
            
            # 计算平均值
            avg_episode = {
                "episode": episode_num,
                "avg_ave_topo_diff": statistics.mean(topo_diffs),
                "avg_ave_slot_collision": statistics.mean(slot_collisions),
                "avg_ave_Throughput": statistics.mean(throughputs),
            }
            
            processed_data.append(avg_episode)
        
        # 3. 保存处理后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        print(f"处理完成！共处理了 {len(processed_data)} 个episode")
        print(f"结果已保存到: {output_file}")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except KeyError as e:
        print(f"错误：JSON格式不正确，缺少必要的键: {e}")
    except json.JSONDecodeError:
        print(f"错误：文件 {input_file} 不是有效的JSON格式")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 处理数据
    # process_dqn_records("training_data_history.json", "episode_averages.json")
    # process_no_reward_records("training_data_history_fixPrd_TPYC.json", "episode_averages_fixPrd_TPYC.json")
    process_no_reward_records("training_data_history_fixPrd_DRAND.json", "episode_averages_fixPrd_DRAND.json")