import csv
import os
import glob

def process_csv_file(file_path):
    """
    处理单个CSV文件，将X/Y/Z列四舍五入取整，生成新文件
    
    参数:
        file_path: 原始CSV文件的路径
    """
    # 构建新文件名：原名称_int.csv
    file_dir, file_name = os.path.split(file_path)
    name_without_ext, ext = os.path.splitext(file_name)
    new_file_name = f"{name_without_ext}_int.csv"
    new_file_path = os.path.join(file_dir, new_file_name)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(new_file_path, 'w', newline='', encoding='utf-8') as outfile:
            
            # 读取并直接写入前两行（注释和表头）
            header1 = infile.readline()
            header2 = infile.readline()
            outfile.write(header1)
            outfile.write(header2)
            
            # 创建CSV读取器和写入器
            csv_reader = csv.reader(infile)
            csv_writer = csv.writer(outfile)
            
            # 处理每一行数据
            for row in csv_reader:
                if len(row) >= 5:  # 确保行有足够的列（Time,Node,X,Y,Z）
                    # 保留前两列（Time, Node）不变
                    new_row = row[:2]
                    # 对X、Y、Z列进行四舍五入取整
                    for col in row[2:5]:
                        try:
                            # 转换为浮点数后四舍五入，再转为整数
                            rounded_val = round(float(col))
                            new_row.append(str(rounded_val))
                        except ValueError:
                            # 如果转换失败，保留原始值
                            new_row.append(col)
                    # 写入处理后的行
                    csv_writer.writerow(new_row)
                else:
                    # 列数不足时直接写入原始行
                    csv_writer.writerow(row)
        
        print(f"成功处理文件：{file_path}")
        print(f"生成新文件：{new_file_path}")
        
    except Exception as e:
        print(f"处理文件 {file_path} 时出错：{str(e)}")

def process_all_csv_in_folder(folder_path='.', file_pattern='*.csv'):
    """
    处理指定文件夹下的所有CSV文件
    
    参数:
        folder_path: 文件夹路径，默认当前目录
        file_pattern: 文件匹配模式，默认所有.csv文件
    """
    # 获取所有符合条件的CSV文件路径
    csv_files = glob.glob(os.path.join(folder_path, file_pattern))
    
    if not csv_files:
        print(f"在 {folder_path} 目录下未找到CSV文件")
        return
    
    # 逐个处理文件
    for csv_file in csv_files:
        process_csv_file(csv_file)

# 主程序入口
if __name__ == "__main__":
    # 处理当前目录下的所有CSV文件
    # process_all_csv_in_folder()
    
    # 如果只想处理单个文件，取消下面的注释并替换为你的文件路径
    process_csv_file("node_position_1s_5s_5_0.7.csv")