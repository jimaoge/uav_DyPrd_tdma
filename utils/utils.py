import json
import numpy as np  # 你的代码里肯定已经导入了，不用重复导
import os

# ========== 新增：定义支持numpy类型的JSON编码器 ==========
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # 处理numpy整数类型 (int64/int32/uint8等)
        if isinstance(obj, np.integer):
            return int(obj)
        # 处理numpy浮点类型 (float64/float32等)
        elif isinstance(obj, np.floating):
            return float(obj)
        # 处理numpy布尔类型
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # 处理numpy数组 (可选，你的代码里已经用tolist了)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # 默认处理原生类型
        return super().default(obj)