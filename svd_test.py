import numpy as np
import pandas as pd


logdir = r'G:\FL\By-FL\logs\2022-06-19\22.17.04\euclidean_matrix'
features = pd.read_csv(logdir + '\euc_round85.csv', header=None)
A = np.array(features)


U, sigma, VT = np.linalg.svd(A)

Sigma = np.diag(sigma)

evalues_path = logdir + '/round85_U.csv'
pd.DataFrame(data=U).to_csv(evalues_path, index=False, header=False)

evalues_path = logdir + '/round85_Sigma.csv'
pd.DataFrame(data=Sigma).to_csv(evalues_path, index=False, header=False)

evalues_path = logdir + '/round85_VT.csv'
pd.DataFrame(data=VT).to_csv(evalues_path, index=False, header=False)


print(U)
print(sigma)
print(VT)

