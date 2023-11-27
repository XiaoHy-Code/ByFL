import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import warnings

warnings.filterwarnings('ignore')

folder_str = r'G:\FL\By-FL\logs\2022-06-20\15.42.28\visualize_TSNE_2D'
folder = pathlib.Path.cwd().parent.joinpath(folder_str)

createVars = locals()  # 以字典类型返回当前位置所有局部变量，后续DataFrame
for fp in folder.iterdir():  # 迭代文件夹
    if fp.match('*.csv'):  # re正则匹配判断文件夹里是否有csv文件
        varname = fp.parts[-1].split('.')[0]  # 按照‘.’的方式切割，取-1，得到csv文件的名字
        # createVars[varname] = pd.read_csv(fp)#添加文件，转为pandas的DataFrame
        # print(varname)#打印文件名
        features = pd.read_csv(fp, header=None)
        features = np.array(features)
        x = features[:, 0]
        y = features[:, 1]
        clients = np.array(range(0, 100))

        fig = plt.figure(figsize=(6, 6))  # 创建画布
        plt.scatter(x, y)
        # plt.plot(x, y, linestyle=' ', marker='o', markersize='4')
        # X轴范围
        # x_min = min(list(x))
        # x_max = max(list(x))
        # x_scal = (x_max - x_min) * 0.1
        # plt.xlim(x_min - x_scal, x_max + x_scal)  # X轴的起点和终点
        # # Y轴范围
        # y_min = min(list(y)) - 50
        # y_max = max(list(y)) + 50
        # y_scal = (y_max - y_min) * 0.01
        # plt.ylim(y_min - y_scal, y_max + y_scal)  # Y轴的起点和终点
        # X轴刻度
        # plt.xticks(np.arange(x_min-1, x_max+1, 100))
        # # X轴刻度
        # plt.yticks(np.arange(6e9, 7e9 + 1e8, 1e8))
        # 标题
        plt.title(varname)
        # 网格线
        plt.grid(axis='both')  # axis: 'both','x','y'
        plt.show()
        fig.savefig(folder_str + '/' + varname + '.png')

# features = pd.read_csv('./logs/2022-06-01/22.14.40/visualize_TSNE/TSNE_round85.csv',header=None)
# features = np.array(features)
#
# x = features[:, 0]
# y = features[:, 1]
# clients = np.array(range(0, 100))
#
# fig = plt.figure()  # 创建画布
# plt.scatter(x, y)
# # plt.plot(x, y, linestyle=' ', marker='o', markersize='4')
# # X轴范围
# plt.xlim(-300, 300)  # X轴的起点和终点
# # Y轴范围
# plt.ylim(-200, 200)  # Y轴的起点和终点
# # X轴刻度
# plt.xticks(np.arange(-300, 301, 100))
# # # X轴刻度
# # plt.yticks(np.arange(6e9, 7e9 + 1e8, 1e8))
# # 标题
# plt.title("2D-TSNE")
# # 网格线
# plt.grid(axis='both')  # axis: 'both','x','y'
# plt.show()
# fig.savefig('./logs/2022-06-01/22.14.40/visualize_TSNE/TSNE_round85.png')
