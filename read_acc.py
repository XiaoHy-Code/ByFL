import pandas as pd


log_file = "path/to/your/log/file.log"  # 指定你的log文件路径
keyword = "Server Defense accuracy: "  # 指定你要匹配的关键字


# 读取log文件为DataFrame
df = pd.read_csv(log_file, sep='\n', header=None, names=['line'])

# 使用str.contains函数提取包含关键字的行
mask = df['line'].str.contains(keyword)
df_filtered = df[mask]

# 使用正则表达式提取数字并保存到list中
import re
accuracy_list = []
pattern = r'\d+\.\d+'
for line in df_filtered['line']:
    match = re.search(pattern, line)
    if match:
        accuracy_list.append(float(match.group()))

# 将list保存到csv文件中
df_output = pd.DataFrame({'accuracy': accuracy_list})
df_output.to_csv('accuracy.csv', index=False)

import re
import csv



# 用于存储提取出的数字
accuracy_list = []

# 打开log文件
with open(log_file, 'r') as f:
    # 逐行读取文件
    for line in f:
        # 判断该行是否包含关键字
        if keyword in line:
            # 使用正则表达式匹配数字
            accuracy = re.findall(r'\d+\.\d+', line)
            # 将匹配到的数字添加到列表中
            accuracy_list.append(float(accuracy[0]))

# 将列表写入CSV文件
with open('accuracy.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['accuracy'])
    writer.writerows([[a] for a in accuracy_list])