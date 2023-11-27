from sklearn.cluster import Birch, KMeans
import pandas as pd
import numpy as np

features = pd.read_csv(r'G:\FL\By-FL\PCA_verify\sketch_map\sample2.csv', header=None)
features = np.array(features)

km = KMeans(n_clusters=2)
km.fit(features)

result = km.labels_
cluster1 = []
cluster2 = []
for id, value in enumerate(result):

    if value == 0:
        cluster1.append(features[id])
    elif value == 1:
        cluster2.append(features[id])

cluster1_path = r'G:\FL\By-FL\PCA_verify\sketch_map' + '/k2_cluster1.csv'
pd.DataFrame(data=cluster1).to_csv(cluster1_path, index=False, header=False)

cluster2_path = r'G:\FL\By-FL\PCA_verify\sketch_map' + '/k2_cluster2.csv'
pd.DataFrame(data=cluster2).to_csv(cluster2_path, index=False, header=False)

print(result)
