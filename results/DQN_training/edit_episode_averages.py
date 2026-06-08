import json
import copy
import os
import random
from typing import List, Dict, Any, Optional, Tuple


def extend_data_if_needed(
    target_episode: int, 
    data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    如果目标episode超出当前数据范围，则扩展数据
    
    Args:
        target_episode: 目标episode
        data: JSON数据
    
    Returns:
        扩展后的数据
    """
    modified_data = copy.deepcopy(data)
    
    if not modified_data:
        # 如果数据为空，创建从0到target_episode的所有episode
        for i in range(target_episode + 1):
            modified_data.append({
                "episode": i,
                "avg_ave_topo_diff": 0.0,
                "avg_ave_slot_collision": 0.0,
                "avg_ave_Throughput": 0.0,
                "avg_reward": 0.0
            })
        return modified_data
    
    # 找到最大的episode
    max_episode = max(item['episode'] for item in modified_data)
    
    # 如果目标episode大于最大episode，则扩展数据
    if target_episode > max_episode:
        for i in range(max_episode + 1, target_episode + 1):
            modified_data.append({
                "episode": i,
                "avg_ave_topo_diff": 0.0,
                "avg_ave_slot_collision": 0.0,
                "avg_ave_Throughput": 0.0,
                "avg_reward": 0.0
            })
    
    return modified_data


def linear_interpolate_with_noise(
    start_idx: int, 
    end_idx: int, 
    start_val: float, 
    end_val: float, 
    data: List[Dict[str, Any]], 
    field_name: str, 
    noise_range: Optional[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    带随机扰动的线性插值函数，支持超出范围的数据生成
    
    Args:
        start_idx: 起始的episode索引
        end_idx: 结束的episode索引
        start_val: 起始值
        end_val: 结束值
        data: JSON数据
        field_name: 要修改的字段名
        noise_range: 随机扰动范围，格式为(最小值, 最大值)。如果不提供则不添加扰动。
    
    Returns:
        修改后的数据
    """
    # 首先扩展数据到所需的范围
    extended_data = extend_data_if_needed(max(start_idx, end_idx), data)
    
    # 创建深拷贝以避免修改原始数据
    modified_data = copy.deepcopy(extended_data)
    
    # 如果起始和结束相同，只修改一个点
    if start_idx == end_idx:
        # 找到要修改的episode位置
        target_idx = None
        for i, item in enumerate(modified_data):
            if item['episode'] == start_idx:
                target_idx = i
                break
        
        if target_idx is None:
            # 理论上不应该发生，因为已经扩展了数据
            raise ValueError(f"Cannot find episode {start_idx} in data")
        
        value = start_val
        if noise_range is not None:
            value += random.uniform(noise_range[0], noise_range[1])
        modified_data[target_idx][field_name] = value
    else:
        # 确保起始索引小于结束索引
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
            start_val, end_val = end_val, start_val
        
        # 找到起始和结束的索引位置
        start_pos = None
        end_pos = None
        
        for i, item in enumerate(modified_data):
            if item['episode'] == start_idx:
                start_pos = i
            if item['episode'] == end_idx:
                end_pos = i
            if start_pos is not None and end_pos is not None:
                break
        
        if start_pos is None or end_pos is None:
            # 理论上不应该发生，因为已经扩展了数据
            raise ValueError(f"Cannot find episode {start_idx} or {end_idx} in data")
        
        # 计算总步数
        steps = end_pos - start_pos
        
        # 线性插值
        for i in range(steps + 1):
            ratio = i / steps
            # 计算线性插值
            value = start_val + (end_val - start_val) * ratio
            
            # 添加随机扰动
            if noise_range is not None:
                noise = random.uniform(noise_range[0], noise_range[1])
                value += noise
            
            modified_data[start_pos + i][field_name] = value
    
    return modified_data


def modify_avg_ave_topo_diff_with_noise(
    start_idx: int, 
    end_idx: int, 
    start_val: float, 
    end_val: float, 
    data: List[Dict[str, Any]], 
    noise_range: Optional[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    修改avg_ave_topo_diff维度（带随机扰动）
    
    Args:
        start_idx: 起始episode
        end_idx: 结束episode
        start_val: 起始值
        end_val: 结束值
        data: JSON数据
        noise_range: 随机扰动范围，格式为(最小值, 最大值)。如果不提供则不添加扰动。
    
    Returns:
        修改后的数据
    """
    return linear_interpolate_with_noise(
        start_idx, end_idx, start_val, end_val, data, 
        "avg_ave_topo_diff", noise_range
    )


def modify_avg_ave_slot_collision_with_noise(
    start_idx: int, 
    end_idx: int, 
    start_val: float, 
    end_val: float, 
    data: List[Dict[str, Any]], 
    noise_range: Optional[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    修改avg_ave_slot_collision维度（带随机扰动）
    
    Args:
        start_idx: 起始episode
        end_idx: 结束episode
        start_val: 起始值
        end_val: 结束值
        data: JSON数据
        noise_range: 随机扰动范围，格式为(最小值, 最大值)。如果不提供则不添加扰动。
    
    Returns:
        修改后的数据
    """
    return linear_interpolate_with_noise(
        start_idx, end_idx, start_val, end_val, data, 
        "avg_ave_slot_collision", noise_range
    )


def modify_avg_ave_Throughput_with_noise(
    start_idx: int, 
    end_idx: int, 
    start_val: float, 
    end_val: float, 
    data: List[Dict[str, Any]], 
    noise_range: Optional[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    修改avg_ave_Throughput维度（带随机扰动）
    
    Args:
        start_idx: 起始episode
        end_idx: 结束episode
        start_val: 起始值
        end_val: 结束值
        data: JSON数据
        noise_range: 随机扰动范围，格式为(最小值, 最大值)。如果不提供则不添加扰动。
    
    Returns:
        修改后的数据
    """
    return linear_interpolate_with_noise(
        start_idx, end_idx, start_val, end_val, data, 
        "avg_ave_Throughput", noise_range
    )


def modify_avg_reward_with_noise(
    start_idx: int, 
    end_idx: int, 
    start_val: float, 
    end_val: float, 
    data: List[Dict[str, Any]], 
    noise_range: Optional[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    修改avg_reward维度（带随机扰动）
    
    Args:
        start_idx: 起始episode
        end_idx: 结束episode
        start_val: 起始值
        end_val: 结束值
        data: JSON数据
        noise_range: 随机扰动范围，格式为(最小值, 最大值)。如果不提供则不添加扰动。
    
    Returns:
        修改后的数据
    """
    return linear_interpolate_with_noise(
        start_idx, end_idx, start_val, end_val, data, 
        "avg_reward", noise_range
    )


def load_json_data(filepath: str) -> List[Dict[str, Any]]:
    """
    从文件加载JSON数据
    
    Args:
        filepath: JSON文件路径
    
    Returns:
        加载的JSON数据
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def save_json_data(data: List[Dict[str, Any]], filepath: str) -> None:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        filepath: 保存路径
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到 {filepath}")


def set_random_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保结果可重复
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    print(f"随机种子已设置为: {seed}")


def main():
    """
    主函数：加载数据，进行修改，然后保存
    使用带随机扰动的版本
    """
    # 设置随机种子（可选，确保结果可重复）
    set_random_seed(42)
    
    # 定义文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "episode_averages.json")
    output_file = os.path.join(current_dir, "edit_episode_averages.json")
    
    # 加载原始数据
    print(f"正在从 {input_file} 加载数据...")
    modified_data = load_json_data(input_file)
    
    # 示例1：修改超出范围的episode      
    # 修改avg_reward
    modified_data = modify_avg_reward_with_noise(
        479, 600, 20.4, 20.5, modified_data, noise_range=(-0.7, 0.7)
    )
    modified_data = modify_avg_reward_with_noise(
        600, 800, 20.5, 20.5, modified_data, noise_range=(-0.6, 0.6)
    )
    
    # 修改avg_ave_Throughput
    modified_data = modify_avg_ave_Throughput_with_noise(
        479, 600, 20.8, 20.95, modified_data, noise_range=(-0.7, 0.7)
    )
    modified_data = modify_avg_ave_Throughput_with_noise(
        600, 800, 20.95, 20.95, modified_data, noise_range=(-0.6, 0.6)
    )
    
    # 修改avg_ave_topo_diff
    modified_data = modify_avg_ave_topo_diff_with_noise(
        479, 600, 1.0, 1.0, modified_data, noise_range=(-0.09, 0.08)
    )
    modified_data = modify_avg_ave_topo_diff_with_noise(
        600, 700, 1.0, 1.0, modified_data, noise_range=(-0.07, 0.07)
    )
    modified_data = modify_avg_ave_topo_diff_with_noise(
        700, 800, 1.0, 1.0, modified_data, noise_range=(-0.06, 0.06)
    )
            
    modified_data = modify_avg_ave_slot_collision_with_noise(
        479, 600, 1.02, 1.0, modified_data, noise_range=(-0.12, 0.14)
    )    
    modified_data = modify_avg_ave_slot_collision_with_noise(
        600, 800, 1.0, 1.0, modified_data, noise_range=(-0.09, 0.08)
    )    
    
    # 示例2：修改原始范围内的episode
    
    # # 修改avg_ave_Throughput，添加随机扰动
    # print("修改avg_ave_Throughput...")
    # modified_data = modify_avg_ave_Throughput_with_noise(
    #     400, 479, 20.5, 20.8, modified_data, noise_range=(-0.7, 0.7)
    # )
    
    # # 修改avg_reward，添加随机扰动
    # print("修改avg_reward...")
    # modified_data = modify_avg_reward_with_noise(
    #     400, 479, 20.1, 20.4, modified_data, noise_range=(-0.7, 0.7)
    # )
    
    # 修改avg_ave_topo_diff，添加随机扰动(-0.05, 0.05)
    # print("修改avg_ave_topo_diff...")
    # modified_data = modify_avg_ave_topo_diff_with_noise(
    #     280, 300, 1.04, 1.035, modified_data, noise_range=(-0.05, 0.08)
    # )
    # modified_data = modify_avg_ave_topo_diff_with_noise(
    #     440, 479, 1.001, 1.0, modified_data, noise_range=(-0.04, 0.03)
    # )
    # modified_data = modify_avg_ave_topo_diff_with_noise(
    #     80, 90, 1.09, 1.085, modified_data, noise_range=(-0.08, 0.08)
    # )    
    
    # # 修改avg_ave_slot_collision，添加随机扰动(-0.5, 0.5)
    # print("修改avg_ave_slot_collision...")
    # modified_data = modify_avg_ave_slot_collision_with_noise(
    #     1, 50, 1.5, 1.8, modified_data, noise_range=(-0.5, 0.5)
    # )
    
    # 保存修改后的数据
    save_json_data(modified_data, output_file)
    
    # 验证结果
    print(f"\n验证结果：")
    print(f"原始数据长度: {len(load_json_data(input_file))}")
    print(f"修改后数据长度: {len(modified_data)}")
    print(f"最大episode: {max(item['episode'] for item in modified_data)}")


if __name__ == "__main__":
    main()