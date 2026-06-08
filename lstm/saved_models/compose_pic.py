from PIL import Image
import os

# -------------------------- 配置项 --------------------------
img_prefix = "training_history_node_"  # 图片前缀
img_suffix = ".png"                    # 图片后缀
n_cols = 3                             # 列数
n_rows = 3                             # 行数
target_size = (400, 300)               # 每张小图统一缩放到的尺寸（可改）
save_name = "training_history_grid.png" # 输出文件名
# -----------------------------------------------------------

# 读取9张图片
images = []
for i in range(9):
    img_path = f"{img_prefix}{i}{img_suffix}"
    if not os.path.exists(img_path):
        print(f"缺失文件: {img_path}")
        exit()
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    images.append(img)

# 计算大图尺寸
w, h = target_size
total_width = n_cols * w
total_height = n_rows * h

# 创建空白画布
grid_img = Image.new("RGB", (total_width, total_height), color=(255,255,255))

# 粘贴图片
idx = 0
for row in range(n_rows):
    for col in range(n_cols):
        grid_img.paste(images[idx], (col * w, row * h))
        idx += 1

# 保存
grid_img.save(save_name)
print(f"九宫格已生成: {save_name}")