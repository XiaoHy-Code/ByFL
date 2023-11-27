
import pandas as pd
import numpy as np
from sklearn.cluster import Birch


features = pd.read_csv(r'H:\By-FL\clustering\r65_grad_data\n5_r65.csv', header=None)
test_data = np.array(features)


y_pred = Birch(n_clusters=2).fit(test_data)

result = y_pred.labels_

clients0 = []
clients1 = []

for id, value in enumerate(result):
    clients0.append('client{}'.format(id)) if value == 0 else clients1.append('client{}'.format(id))

if len(clients0) >= len(clients1):
    benign_client = clients0
    malicious_clients = clients1
else:
    benign_client = clients1
    malicious_clients = clients0


print()