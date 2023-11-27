import pandas as pd
import numpy as np
import pathlib
import warnings

from PCA_verify.my_PCA import PCA_skl
from utils import mkdirs

warnings.filterwarnings('ignore')

folder_str = r'G:\FL\By-FL\logs\2022-07-14\11.30.56\gradient'
folder = pathlib.Path.cwd().parent.joinpath(folder_str)

path_str = r'G:\FL\By-FL\PCA_verify'
mkdirs(path_str + '/Reducing_grad/')
createVars = locals()  # 以字典类型返回当前位置所有局部变量，后续DataFrame

n_list = [100, 50, 20, 10, 5, 2, 1]

for n in n_list:
    for fp in folder.iterdir():  # 迭代文件夹
        if fp.match('*.csv'):  # re正则匹配判断文件夹里是否有csv文件
            varname = fp.parts[-1].split('.')[0]  # 按照‘.’的方式切割，取-1，得到csv文件的名字
            # createVars[varname] = pd.read_csv(fp)#添加文件，转为pandas的DataFrame
            print(varname)  # 打印文件名
            grad = pd.read_csv(fp, header=None)
            grad = np.array(grad)


            r = varname.split('grad_round')[1]

            param_, _ = PCA_skl(grad, n)

            file_name = 'n' + str(n) + '_' + 'r' + str(r) + '.csv'
            print(file_name)
            pca_path = path_str + '/Reducing_grad/' + file_name
            pd.DataFrame(data=param_).to_csv(pca_path, index=False, header=False)
