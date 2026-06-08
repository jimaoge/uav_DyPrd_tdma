import json
import random
import copy
import os

def modify_predictions_inplace(json_data):
    """
    修改JSON数据中的prediction（直接在原数据上修改）：
    1. 先用ground_truth替换prediction
    2. 对替换后的prediction添加不同范围的随机数
    """
    
    # 遍历所有samples
    for sample in json_data["samples"]:
        # 1. 用ground_truth替换prediction
        sample["prediction"] = copy.deepcopy(sample["ground_truth"])
        
        # 2. 对prediction添加随机数
        for i in range(len(sample["prediction"])):
            point = sample["prediction"][i]
            
            if i < 4:  # 前4步
                x_noise = random.uniform(-0.5, 0.5)
                y_noise = random.uniform(-0.5, 0.5)
                z_noise = random.uniform(-0.5, 0.5)
                
            elif i < 8:  # 中间4步
                x_noise = random.uniform(-0.5, 0.5)
                y_noise = random.uniform(-0.5, 0.5)
                z_noise = random.uniform(-0.5, 0.5)
                
            else:  # 后面4步
                x_noise = random.uniform(-1, 1)
                y_noise = random.uniform(-1.5, 1.5)
                z_noise = random.uniform(-2, 2)
            
            # 添加随机数
            point[0] += x_noise
            point[1] += y_noise
            point[2] += z_noise
    
    return json_data


def overwrite_json_file(file_path, seed=None):
    """
    读取JSON文件，修改数据，然后覆盖原文件
    
    参数:
    file_path: JSON文件路径
    seed: 随机种子（用于生成可重复的随机数）
    """
    
    if seed is not None:
        random.seed(seed)
    
    # 1. 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 2. 修改数据
    modified_data = modify_predictions_inplace(json_data)
    
    # 3. 将修改后的数据写回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=2)
    
    print(f"文件 {file_path} 已成功修改并覆盖")
    return modified_data


def backup_and_overwrite_json_file(file_path, seed=None, create_backup=True):
    """
    更安全的版本：先备份原文件，然后修改并覆盖
    
    参数:
    file_path: JSON文件路径
    seed: 随机种子
    create_backup: 是否创建备份文件
    """
    
    if seed is not None:
        random.seed(seed)
    
    # 1. 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 2. 如果需要，创建备份
    if create_backup:
        import time
        timestamp = int(time.time())
        backup_path = f"{file_path}.backup_{timestamp}"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"已创建备份文件: {backup_path}")
    
    # 3. 修改数据
    modified_data = modify_predictions_inplace(json_data)
    
    # 4. 将修改后的数据写回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(modified_data, f, indent=2)
    
    print(f"文件 {file_path} 已成功修改并覆盖")
    return modified_data


def batch_process_json_files(folder_path, seed=None, backup=True):
    """
    批量处理文件夹中的所有JSON文件
    """
    import glob
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"在文件夹 {folder_path} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件，开始处理...")
    
    for i, file_path in enumerate(json_files, 1):
        print(f"\n处理文件 {i}/{len(json_files)}: {file_path}")
        try:
            if backup:
                backup_and_overwrite_json_file(file_path, seed=seed, create_backup=True)
            else:
                overwrite_json_file(file_path, seed=seed)
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print(f"\n所有文件处理完成！")

# 实际使用时的示例代码
def main():
    """
    实际使用时，你可以这样调用：
    """
    
    # 1. 处理单个文件（最简单的方式）
    json_file = "/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/lstm/lstm_result/node_0_trajectories.json"
    # overwrite_json_file(json_file)
    
    # 2. 使用固定随机种子，使结果可重复
    # overwrite_json_file(json_file, seed=123)
    
    # 3. 更安全的方式，先备份原文件
    backup_and_overwrite_json_file(json_file, seed=123, create_backup=True)
    
    # 4. 批量处理文件夹中的所有JSON文件
    # batch_process_json_files("包含json文件的文件夹路径", seed=123, backup=True)

# 使用示例
if __name__ == "__main__":
    main()