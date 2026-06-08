# 导入所需库
import json
import matplotlib.pyplot as plt

# 1. 读取JSON文件
with open("DyPrd_cmp.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 2. 提取绘图数据（速度、平均DyPrd值）
performance_list = json_data["performance_data"]
speed_list = [item["speed"] for item in performance_list]       # 横轴数据
dyprd_list = [item["average_DyPrd"] for item in performance_list] # 纵轴数据

# 3. 创建柱状图
plt.bar(speed_list, dyprd_list, width=3)  # 绘制柱状图，设置颜色/宽度

# 4. 设置坐标轴标签（严格按照你的要求，全英文）
plt.xlabel("Node Average Speed(m/s)")  # 横轴标签
plt.ylabel("Average DyPrd(s)")         # 纵轴标签
plt.grid(axis="y", linestyle="--", alpha=0.7)

filename = f'DyPrd_comparison.svg'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"图表已保存为: '{filename}'")