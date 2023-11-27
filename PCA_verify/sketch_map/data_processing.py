import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv(r'G:\FL\By-FL\PCA_verify\sketch_map\true_cluster2.csv')

col_1 = data["x"]
data_x = np.array(col_1)

avg_x = sum(data_x) / len(data_x)
data_x_new = avg_x + (data_x - avg_x) * 1.2 + 30


col_1 = data["y"]
data_y = np.array(col_1)
avg_y = sum(data_y) / len(data_y)
data_y_new = avg_y + (data_y - avg_y) * 1.2 - 30

DataSet = list(zip(data_x_new, data_y_new))
df = pd.DataFrame(data=DataSet, columns=['x', 'y'])
df.to_csv(r'G:\FL\By-FL\PCA_verify\sketch_map\new_cluster2.csv', index=False, header=False)