import pandas as pd
import numpy as np
import pathlib
import warnings

from utils import *

warnings.filterwarnings('ignore')

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# folder_str = r'G:\FL\By-FL\PCA_verify\Reducing_grad'
folder_str = r'G:\FL\By-FL\logs\2022-07-14\11.30.56\gradient'
folder = pathlib.Path.cwd().parent.joinpath(folder_str)

path_str = r'G:\FL\By-FL\PCA_verify'

for fp in folder.iterdir():  # 迭代文件夹
    if fp.match('*.csv'):  # re正则匹配判断文件夹里是否有csv文件
        varname = fp.parts[-1].split('.')[0]  # 按照‘.’的方式切割，取-1，得到csv文件的名字
        # createVars[varname] = pd.read_csv(fp)#添加文件，转为pandas的DataFrame
        print(varname)  # 打印文件名
        # n = varname.split('n')[1].split('_')[0]
        # print(n)
        r = varname.split('grad_round')[1]
        print(r)

        grad = pd.read_csv(fp, header=None)
        grad = np.array(grad)
        n = grad.shape[1]
        print(n)
        cosine_matrix = cosine_clients(grad, dev)
        euclidean_matrix = euclidean_clients(grad, dev)

        mkdirs(path_str + '/cosine_matrix/')
        cos_name = '/cos_' + 'r' + r + '_' + 'n' + n + '.csv'
        cosine_path = path_str + '/cosine_matrix/' + cos_name
        pd.DataFrame(data=cosine_matrix).to_csv(cosine_path, index=False, header=False)

        mkdirs(path_str + '/euclidean_matrix/')
        euc_name = '/euc_' + 'r' + r + '_' + 'n' + n + '.csv'
        euclidean_path = path_str + '/euclidean_matrix/' + euc_name
        pd.DataFrame(data=euclidean_matrix).to_csv(euclidean_path, index=False, header=False)
        print(cos_name + ' ' + euc_name + " finished")

# for fp in folder.iterdir():  # 迭代文件夹
#     if fp.match('*.csv'):  # re正则匹配判断文件夹里是否有csv文件
#         varname = fp.parts[-1].split('.')[0]  # 按照‘.’的方式切割，取-1，得到csv文件的名字
#         # createVars[varname] = pd.read_csv(fp)#添加文件，转为pandas的DataFrame
#         print(varname)  # 打印文件名
#         n = varname.split('n')[1].split('_')[0]
#         print(n)
#         r = varname.split('r')[1]
#         print(r)
#
#         grad = pd.read_csv(fp, header=None)
#         grad = np.array(grad)
#         cosine_matrix = cosine_clients(grad, dev)
#         euclidean_matrix = euclidean_clients(grad, dev)
#
#         mkdirs(path_str + '/cosine_matrix/')
#         cos_name = '/cos_' + 'r' + r + '_' + 'n' + n + '.csv'
#         cosine_path = path_str + '/cosine_matrix/' + cos_name
#         pd.DataFrame(data=cosine_matrix).to_csv(cosine_path, index=False, header=False)
#
#         mkdirs(path_str + '/euclidean_matrix/')
#         euc_name = '/euc_' + 'r' + r + '_' + 'n' + n + '.csv'
#         euclidean_path = path_str + '/euclidean_matrix/' + euc_name
#         pd.DataFrame(data=euclidean_matrix).to_csv(euclidean_path, index=False, header=False)
#         print(cos_name + ' ' + euc_name + " finished")