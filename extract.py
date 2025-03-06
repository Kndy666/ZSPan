import os
import glob
import pandas as pd

# 指定 CSV 文件所在的文件夹路径
folder_path = "/home/ctz/ZSpan_test/ZSPan/loss_data/"

# 获取所有 CSV 文件路径（假设扩展名为 .csv）
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

merged = None

for file in csv_files:
    # 读取整个文件，用于调试查看实际列名
    df_temp = pd.read_csv(file)
    print("文件 {} 的列名: {}".format(file, df_temp.columns.tolist()))
    
    # 直接根据列的索引位置提取第一列（Step）和第二列（对应数值）
    df = pd.read_csv(file, usecols=[0, 1])
    # 重命名列，确保第一列为 "Step"
    df.columns = ["Step", "value"]
    
    # 将 "Step" 设为索引，以便后续合并
    df = df.set_index("Step")
    
    # 用文件名（去除扩展名）作为本文件对应的列名
    col_name = os.path.splitext(os.path.basename(file))[0]
    df = df.rename(columns={"value": col_name})
    
    # 按 Step 对齐合并数据（外连接，确保所有 Step 均出现）
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, left_index=True, right_index=True, how="outer")

# 重置索引，将 Step 变成 DataFrame 的一列
merged = merged.reset_index()

# 删除 "Step" 列
merged = merged.drop(columns=["Step"])

# 指定列的顺序
desired_columns = ["fullratio_loss_0.1","fullratio_loss_0.2","fullratio_loss_0.6","fullratio_loss_0.8","fullratio_loss_0.9"]

# 按照所需顺序排列列
merged = merged[desired_columns]

# 保存合并结果到新的 CSV 文件
output_file = os.path.join(folder_path, "merged_steps_restructured.csv")
merged.to_csv(output_file, index=False)

print(f"合并后的 CSV 文件已保存：{output_file}")
