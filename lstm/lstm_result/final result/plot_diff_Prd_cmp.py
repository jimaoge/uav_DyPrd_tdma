# 导入必要的库
import matplotlib.pyplot as plt
import json

# ---------------------- 数据准备 ----------------------
# 读取JSON文件
with open('/root/autodl-tmp/lstm_UAV_predict/uav_DyPrd_tdma/lstm/lstm_result/final result/diff_Prd_cmp.json', 'r', encoding='utf-8') as f:
    performance_json = json.load(f)

# 提取数据：横轴Prd、左纵轴average_diff、右纵轴accuracy
prd_values = [item["prd"] for item in performance_json["performance_data"]]
avg_diff_values = [item["average_diff"] for item in performance_json["performance_data"]]
accuracy_values = [item["accuracy"] for item in performance_json["performance_data"]]

# 创建画布，设置尺寸
fig, ax1 = plt.subplots(figsize=(8, 5))

# ---------------------- 绘制第一条曲线：Topology Difference（左纵轴） ----------------------
line1 = ax1.plot(prd_values, avg_diff_values, 
                marker='o',        # 圆点标记
                markersize=7,      
                linestyle='-',     
                linewidth=2,       
                color='#1f77b4',   # 深蓝色
                label='Topology Difference')

# 左纵轴配置
ax1.set_xlabel('Prd(s)', fontsize=12, fontweight='medium')
ax1.set_ylabel('Topology Difference', fontsize=12, fontweight='medium')
ax1.set_xlim(0, 12)  # 横轴范围 0~12
ax1.tick_params(axis='both', labelsize=10)
ax1.grid(False)  # 关闭网格

# ---------------------- 绘制第二条曲线：Accuracy（右纵轴） ----------------------
ax2 = ax1.twinx()  # 创建共享横轴的右纵轴
# 将accuracy转换为百分数
accuracy_percent = [acc * 100 for acc in accuracy_values]
line2 = ax2.plot(prd_values, accuracy_percent,
                marker='^',        # 三角标记
                markersize=7,
                linestyle='-',
                linewidth=2,
                color='#d62728',   # 红色
                label='Accuracy')

# 右纵轴配置
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='medium')
ax2.tick_params(axis='both', labelsize=10)
ax2.grid(False)  # 关闭网格

# ---------------------- 图例：垂直居中右侧 ----------------------
lines = line1 + line2
labels = [l.get_label() for l in lines]
# loc='center right' = 图表中间靠右的位置
ax1.legend(lines, labels, fontsize=10, loc='center right')

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('lstm_performance.svg', dpi=300, bbox_inches='tight')
# plt.show()