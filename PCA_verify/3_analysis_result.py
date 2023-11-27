import numpy as np
import pathlib
import pandas as pd


m_client = [0, 3, 8, 12, 15, 17, 26, 32, 48, 49, 55, 57, 60, 62, 63, 72, 77, 83, 97, 98]

#准备工作，准备好三个mask矩阵
vector_m = np.zeros(100)
for m in m_client:
    vector_m[m] = 1
# print(vector_m)
vector_m_T = vector_m.T.reshape(100, 1)
mask_mm = np.multiply(vector_m_T, vector_m)
# print(mask_mm)

vector_b = vector_m - 1
vector_b_T = vector_b.T.reshape(100, 1)
mask_bb = np.multiply(vector_b_T, vector_b) + np.diag(vector_b)
print(mask_bb)

mask_mb = np.ones((100, 100)) - mask_mm - mask_bb + np.diag(vector_b)
print(mask_mb)

bb = sum(sum(mask_bb))
print(bb)

mb = sum(sum(mask_mb))
print(mb)


print("***********************************************************")

folder_str = r'./PCA_verify/cosine_matrix'
folder = pathlib.Path.cwd().parent.joinpath(folder_str)
for fp in folder.iterdir():  # 迭代文件夹
    if fp.match('*.csv'):  # re正则匹配判断文件夹里是否有csv文件
        varname = fp.parts[-1].split('.')[0]  # 按照‘.’的方式切割，取-1，得到csv文件的名字
        # createVars[varname] = pd.read_csv(fp)#添加文件，转为pandas的DataFrame
        print(varname)  # 打印文件名
        euc = pd.read_csv(fp, header=None)
        euc = np.array(euc)
        dis_matix_bb = np.multiply(mask_bb, euc)
        dis_matix_mb = np.multiply(mask_mb, euc)
        avg_dis_bb = sum(sum(dis_matix_bb))/bb #所有良性和良性之间距离的平均
        avg_dis_mb = sum(sum(dis_matix_mb))/mb #所有恶意和良性之间距离的平均
        print("bb: %.4f" % avg_dis_bb)
        print("mb: %.4f" % avg_dis_mb)
        relative_error = (avg_dis_mb-avg_dis_bb)/avg_dis_mb  #这两个距离的相对误差
        print("error: %.4f" % relative_error) #误差越大说明良性和恶意的区分越明显
        print()