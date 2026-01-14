import json
import csv
import numpy as np
from datetime import datetime
from math import sqrt
from typing import Dict, List, Tuple
import argparse
import os

def load_uav_data(csv_path: str) -> Dict[float, Dict[int, List[float]]]:
    """
    加载UAV轨迹数据
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        字典结构: {时间: {节点ID: [x, y, z]}}
    """
    data = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:  # 跳过格式不正确的行
                continue
                
            try:
                time = float(row[0].strip())
                node_id = int(row[1].strip())
                x = float(row[2].strip())
                y = float(row[3].strip())
                z = float(row[4].strip())
            except ValueError:
                continue
                
            if time not in data:
                data[time] = {}
            data[time][node_id] = [x, y, z]
    
    return data

def calculate_distance(pos1: List[float], pos2: List[float]) -> float:
    """
    计算两个节点之间的欧几里得距离
    
    Args:
        pos1: 节点1的坐标 [x, y, z]
        pos2: 节点2的坐标 [x, y, z]
        
    Returns:
        两点之间的距离
    """
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return sqrt(dx*dx + dy*dy + dz*dz)

def generate_adjacency_matrix(data: Dict[float, Dict[int, List[float]]], 
                              comm_range: float, 
                              node_count: int = 9) -> List[Dict]:
    """
    生成邻接矩阵
    
    Args:
        data: UAV轨迹数据
        comm_range: 通信距离
        node_count: 节点数量
        
    Returns:
        包含拓扑信息的列表
    """
    topologies = []
    id_counter = 0
    
    # 按时间排序
    sorted_times = sorted(data.keys())
    
    for time in sorted_times:
        if len(data[time]) != node_count:
            print(f"警告: 时间{time}秒只有{len(data[time])}个节点的数据，跳过此时间点")
            continue
        
        # 初始化邻接矩阵
        adj_matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]
        
        # 计算节点间的连通性
        for i in range(node_count):
            for j in range(node_count):
                if i == j:  # 对角线保持为0
                    continue
                    
                if i in data[time] and j in data[time]:
                    distance = calculate_distance(data[time][i], data[time][j])
                    if distance <= comm_range:
                        adj_matrix[i][j] = 1
        
        # 添加到结果列表
        topologies.append({
            "id": id_counter,
            "time": float(time),
            "one_hop_matrix": adj_matrix
        })
        id_counter += 1
    
    return topologies

def filter_by_time_range(topologies: List[Dict], 
                         start_time: float, 
                         duration: float) -> List[Dict]:
    """
    按时间范围过滤拓扑数据
    
    Args:
        topologies: 拓扑数据列表
        start_time: 起始时间
        duration: 持续时间
        
    Returns:
        过滤后的拓扑数据
    """
    filtered = []
    end_time = start_time + duration
    
    for topology in topologies:
        if start_time <= topology["time"] < end_time:
            filtered.append(topology)
    
    return filtered

def save_to_json(topologies: List[Dict], 
                 output_path: str, 
                 node_count: int = 9,
                 slot_count: int = 9):
    """
    将拓扑数据保存为JSON文件
    
    Args:
        topologies: 拓扑数据列表
        output_path: 输出文件路径
        node_count: 节点数量
        slot_count: 时隙数量
    """
    result = {
        "description": "TDMA网络邻接矩阵数据",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "node_count": node_count,
        "slot_count": slot_count,
        "topologies": topologies
    }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"数据已保存到: {output_path}")
    print(f"包含拓扑数量: {len(topologies)}")

def save_to_json_simple(topologies: List[Dict], 
                       output_path: str, 
                       node_count: int = 9,
                       slot_count: int = 9):
    """
    将拓扑数据保存为JSON文件的简化版本
    手动控制邻接矩阵的格式
    """
    result = {
        "description": "TDMA网络邻接矩阵数据",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "node_count": node_count,
        "slot_count": slot_count,
        "topologies": []
    }
    
    # 手动构建JSON字符串
    json_str = '{\n'
    json_str += f'  "description": "TDMA网络邻接矩阵数据",\n'
    json_str += f'  "created_at": "{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}",\n'
    json_str += f'  "node_count": {node_count},\n'
    json_str += f'  "slot_count": {slot_count},\n'
    json_str += '  "topologies": [\n'
    
    for i, topology in enumerate(topologies):
        json_str += '    {\n'
        json_str += f'      "id": {topology["id"]},\n'
        json_str += f'      "time": {topology["time"]},\n'
        json_str += '      "one_hop_matrix": [\n'
        
        # 添加邻接矩阵，每行格式为：[0, 1, 0, ...]
        matrix = topology["one_hop_matrix"]
        for j, row in enumerate(matrix):
            row_str = '        [' + ', '.join(str(x) for x in row) + ']'
            if j < len(matrix) - 1:
                row_str += ','
            json_str += row_str + '\n'
        
        json_str += '      ]\n'
        json_str += '    }'
        if i < len(topologies) - 1:
            json_str += ','
        json_str += '\n'
    
    json_str += '  ]\n'
    json_str += '}'
    
    with open(output_path, 'w') as f:
        f.write(json_str)
    
    print(f"数据已保存到: {output_path}")
    print(f"包含拓扑数量: {len(topologies)}")

def main():
    parser = argparse.ArgumentParser(description='生成UAV网络邻接矩阵')
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='输入CSV文件路径')
    parser.add_argument('--output', '-o', type=str, default='topology.json',
                       help='输出JSON文件路径，默认为topology.json')
    parser.add_argument('--range', '-r', type=float, default=200.0,
                       help='通信距离（米），默认为200米')
    parser.add_argument('--start', '-s', type=float, default=10.0,
                       help='起始时间（秒），默认为10秒')
    parser.add_argument('--duration', '-d', type=float, default=5.0,
                       help='持续时间（秒），默认为5秒')
    parser.add_argument('--nodes', '-n', type=int, default=9,
                       help='节点数量，默认为9')
    parser.add_argument('--slots', '-t', type=int, default=9,
                       help='时隙数量，默认为9')
    
    args = parser.parse_args()
    
    print(f"参数设置:")
    print(f"  输入文件: {args.input}")
    print(f"  输出文件: {args.output}")
    print(f"  通信距离: {args.range}米")
    print(f"  起始时间: {args.start}秒")
    print(f"  持续时间: {args.duration}秒")
    print(f"  节点数量: {args.nodes}")
    print(f"  时隙数量: {args.slots}")
    print()
  
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return
    
    try:
        # 1. 加载UAV数据
        print("正在加载UAV轨迹数据...")
        uav_data = load_uav_data(args.input)
        print(f"加载完成，共包含 {len(uav_data)} 个时间点")
        
        # 2. 生成所有时间点的邻接矩阵
        print("正在生成邻接矩阵...")
        all_topologies = generate_adjacency_matrix(uav_data, args.range, args.nodes)
        
        # 3. 按时间范围过滤
        print(f"正在过滤时间范围: {args.start}秒 ~ {args.start + args.duration}秒")
        filtered_topologies = filter_by_time_range(all_topologies, args.start, args.duration)
        
        if not filtered_topologies:
            print(f"警告: 在指定时间范围内未找到拓扑数据")
            print(f"可用时间范围: {min(uav_data.keys()):.1f}秒 ~ {max(uav_data.keys()):.1f}秒")
            return
        
        # 4. 保存为JSON文件
        save_to_json_simple(filtered_topologies, args.output, args.nodes, args.slots)
        
        # 5. 打印统计信息
        print()
        print("统计信息:")
        times = [t["time"] for t in filtered_topologies]
        print(f"  时间点数量: {len(filtered_topologies)}")
        print(f"  时间范围: {min(times):.1f}秒 ~ {max(times):.1f}秒")
        print(f"  时间间隔: {times[1] - times[0] if len(times) > 1 else 'N/A'}秒")
        
        # 计算平均连接度
        avg_connections = []
        for topology in filtered_topologies:
            matrix = topology["one_hop_matrix"]
            connections = sum(sum(row) for row in matrix) / 2  # 除以2因为是无向图
            avg_connections.append(connections)
        
        if avg_connections:
            avg_conn = np.mean(avg_connections)
            max_conn = np.max(avg_connections)
            min_conn = np.min(avg_connections)
            print(f"  平均连接数: {avg_conn:.2f}")
            print(f"  最小连接数: {min_conn}")
            print(f"  最大连接数: {max_conn}")
            
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()