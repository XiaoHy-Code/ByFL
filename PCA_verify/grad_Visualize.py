
import pandas as pd
import numpy as np
from utils import mkdirs

path_str = r'G:\FL\By-FL\PCA_verify'
mkdirs(path_str + '/Reducing_grad/')


features = pd.read_csv(r'G:\FL\By-FL\logs\2022-07-14\11.30.56\gradient\grad_round15.csv', header=None)
features = np.array(features)

H, W = features.shape

result = np.zeros((100, 100))


for i in range(100):
    for j in range(100):
        cur_input = features[i, j*785:(j+1)*785]
        cur_output = np.sum(cur_input)
        result[i, j] = cur_output
    print(i)
file_name = 'vis_100.csv'
pca_path = path_str + '/Reducing_grad/' + file_name
pd.DataFrame(data=result).to_csv(pca_path, index=False, header=False)

