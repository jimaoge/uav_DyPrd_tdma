import json

def calculate_avg_DyPrd_per_episode(input_file, output_file):
    """
    计算每个episode的平均DyPrd并保存到JSON文件
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    try:
        # 1. 读取JSON文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 2. 计算每个episode的平均DyPrd
        results = []
        
        for episode_data in data:
            episode_num = episode_data.get("episode")
            steps = episode_data.get("steps", [])
            
            if not steps:  # 如果steps为空，平均值为0
                avg_dyprd = 0
            else:
                # 提取所有DyPrd值
                dyprd_values = [step.get("DyPrd", 0) for step in steps]
                # 计算平均值
                avg_dyprd = sum(dyprd_values) / len(dyprd_values)
            
            # 添加到结果列表
            results.append({
                "episode": episode_num,
                "avg_DyPrd": avg_dyprd
            })
        
        # 3. 保存结果到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"计算完成！结果已保存到: {output_file}")
        print(f"共处理了 {len(results)} 个episode")
        
        # 4. 打印前几个结果作为预览
        print("\n前5个episode的平均DyPrd:")
        for i, result in enumerate(results[:5]):
            print(f"  Episode {result['episode']}: {result['avg_DyPrd']:.4f}")
            
        return results
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误: {input_file} 不是有效的JSON文件")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 输入文件路径
    input_file = "/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/results/DQN_training/training_data_history.json"
    # 输出文件路径
    output_file = "/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/results/DQN_training/average_DyPrd_per_episode.json"
    
    # 计算并保存结果
    calculate_avg_DyPrd_per_episode(input_file, output_file)