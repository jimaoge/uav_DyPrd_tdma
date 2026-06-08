import json
import matplotlib.pyplot as plt

# 1. 读取我们的 JSON 数据文件
with open("diff_speed.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)

# 2. 提取横轴时间、纵轴速度
times = [point["time"] for point in json_data["data_points"]]
speeds = [point["avg_speed"] for point in json_data["data_points"]]

# 3. 字体设置（兼容英文，无乱码）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 4. 绘制图表
plt.plot(times, speeds, color="#1f77b4", linewidth=2, marker="o", markersize=6)
plt.xlabel(json_data["x_label"], fontsize=12)
plt.ylabel(json_data["y_label"], fontsize=12)
plt.xticks(range(0, 160, 10))
plt.grid(alpha=0.3)

# 5. 保存为 SVG 矢量图
plt.savefig("node_speed_curve.svg", format="svg", dpi=300, bbox_inches="tight")
plt.close()
print("SVG image generated: node_speed_curve.svg")