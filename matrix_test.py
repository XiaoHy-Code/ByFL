import numpy as np
import pandas as pd

from utils import mkdirs, computer_defense_acc

clients_in_comm = ['client{}'.format(i) for i in range(0, 100)]

logdir = r'G:\FL\By-FL\logs\2022-06-19\22.17.04\euclidean_matrix'
features = pd.read_csv(logdir + '\euc_round85.csv', header=None)
A = np.array(features)

# A = np.mat("1 2 3; 2 3 4; 5 4 6")  #创建矩阵print("A\n", A)
inverse = np.linalg.inv(A)
print("inverse\n", inverse)
# 单纯的求解矩阵的特征值print("eigenvalues: ", eigenvalues)

malicious_clients = ['client17', 'client72', 'client97', 'client8', 'client32', 'client15', 'client63', 'client57',
                     'client60', 'client83', 'client48', 'client26', 'client12', 'client62', 'client3', 'client49',
                     'client55', 'client77', 'client0', 'client92', 'client34', 'client29', 'client75', 'client13',
                     'client40', 'client85', 'client2', 'client74', 'client69', 'client1']

eigenvalues = np.linalg.eigvals(A)
client_evalues = dict(zip(clients_in_comm, list(eigenvalues)))
client_scores = sorted(client_evalues.items(), key=lambda d: d[1], reverse=False)


d_benign_client = [idx for idx, val in client_scores[:30]]
d_malicious_client = [idx for idx, val in client_scores[len(clients_in_comm) - 30:]]

d_benign_acc = computer_defense_acc(d_benign_client, malicious_clients)
d_malicious_acc = computer_defense_acc(d_malicious_client, malicious_clients)

e_values = np.diag(eigenvalues)
evalues_path = logdir + '/evalues_round85.csv'
pd.DataFrame(data=e_values).to_csv(evalues_path, index=False, header=False)
print("eigenvalues: ", e_values)  # 特征值

eigenvectors = np.linalg.eig(A)[1]
evectors_path = logdir + '/evectors_round85.csv'
pd.DataFrame(data=eigenvectors).to_csv(evectors_path, index=False, header=False)
print("eigenvectors: ", eigenvectors)  # 特征向量
