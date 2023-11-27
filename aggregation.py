import numpy as np
from hdbscan import HDBSCAN
import sklearn.metrics.pairwise as smp
from sklearn.cluster import KMeans, Birch, AgglomerativeClustering
# from torch.distributed.pipeline.sync import copy
import copy
from collections import Counter
from PCA_verify.my_PCA import PCA_skl
from utils import *
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd

# def server_agg(client_params, agg_type, len_malicious_clients,  pca_d, logger):
#
#
#     if agg_type == 'krum' or agg_type == 'multi_krum':
#         print("agg_type: " + "krum/multi_krum")
#         logger.info("agg_type: " + "krum/multi_krum")
#         global_parameters, detect_malicious_client = agg_multi_krum(client_params, len_malicious_clients)
#
#     elif agg_type == 'pca_kmeans':
#         print("agg_type: " + "pca_kmeans")
#         logger.info("agg_type: " + "pca_kmeans")
#
#         global_parameters, detect_malicious_client = agg_pca_kmeans(client_params, pca_d)
#
#     elif agg_type == 'pca_birch':
#         print("agg_type: " + "pca_birch")
#         logger.info("agg_type: " + "pca_birch")
#         global_parameters, detect_malicious_client = agg_pca_birch(client_params, pca_d)
#
#     else:
#         print("agg_type: " + "average")
#         logger.info("agg_type: " + "average")
#
#         global_parameters = agg_average(client_params)
#         detect_malicious_client = {}
#
#     return global_parameters, detect_malicious_client


def agg_average(client_params):
    # 把需要聚合的模型参数，放到一个tensor中
    user_grads = []
    for client in client_params:
        local_parameters = client_params[client]
        user_grads = local_parameters[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_parameters[None, :]), 0)

    # 用torch平均参数
    avg_params = torch.mean(user_grads, dim=0)

    return avg_params


def agg_multi_krum(client_params, num_adv):
    user_grads = []
    clients_in_comm = []
    for client in client_params:
        clients_in_comm.append(client)
        local_parameters = client_params[client]
        user_grads = local_parameters[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_parameters[None, :]), 0)

    euclidean_matrix = euclidean_clients(user_grads)

    # compute scores
    scores = []
    for list in euclidean_matrix:
        client_dis = sorted(list)
        client_dis1 = client_dis[1:len(client_params) - num_adv]
        score = np.sum(np.array(client_dis1))
        scores.append(score)
    client_scores = dict(zip(clients_in_comm, scores))
    client_scores = sorted(client_scores.items(), key=lambda d: d[1], reverse=False)

    benign_client = client_scores[:len(client_params) - num_adv]
    benign_client = [idx for idx, val in benign_client]
    malicious_client = client_scores[len(client_params) - num_adv:]
    malicious_client = [idx for idx, val in malicious_client]

    benign_client_params = {}
    for client in client_params:
        if client in benign_client:
            benign_client_params[client] = client_params[client]

    global_parameters = agg_average(benign_client_params)

    return global_parameters, malicious_client


def agg_auror(client_params, dataset):
    user_grads = []
    for client in client_params:
        local_parameters = client_params[client]

        if dataset == 'cifar10':
            local_parameters = (local_parameters.reshape(-1, 2))[:, 0]
        user_grads = local_parameters[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_parameters[None, :]), 0)

    grad = user_grads.cpu().numpy()
    km = KMeans(n_clusters=2).fit(grad)
    result = km.labels_
    # 1 is the label of malicious clients
    if sum(result) > len(result) / 2:
        result = 1 - result

    return result


def agg_foolsgold(client_params):
    user_grads = []
    clients_in_comm = []
    for client in client_params:
        clients_in_comm.append(client)
        local_parameters = client_params[client]
        user_grads = local_parameters[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_parameters[None, :]), 0)

    # cos_matrix = cosine_clients(user_grads)

    n_clients = user_grads.shape[0]
    # 两两客户端相似度
    cs = smp.cosine_similarity(user_grads) - np.eye(n_clients)

    # 取最大相似度作为自己的相似度
    maxcs = np.max(cs, axis=1)
    # pardoning诚实客户可能会在该方案下受到错误的惩罚。我们引入了一种赦免机制，通过用vi和vj之比重新加权余弦相似性
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)+1e-5) + 0.5)


    # wv[(np.isinf(wv) + wv > 1)] = 1
    # wv[(wv < 0)] = 0

    benign_client_params = {}
    malicious_client = []
    for idx, client in enumerate(client_params):
        if wv[idx] > 0:
        # if client in benign_client:
            benign_client_params[client] = client_params[client]
        else:
            malicious_client.append(client)

    global_parameters = agg_average(benign_client_params)

    return global_parameters, malicious_client

def agg_pca_kmeans(client_params, pca_d, dataset):
    user_grads = []
    for client in client_params:
        local_parameters = client_params[client]
        if dataset == 'cifar10':
            local_parameters = (local_parameters.reshape(-1, 2))[:, 0]
        user_grads = local_parameters[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_parameters[None, :]), 0)

    grad = user_grads.cpu().numpy()
    grad, _ = PCA_skl(grad, pca_d)
    km = KMeans(n_clusters=2).fit(grad)
    result = km.labels_

    # 1 is the label of malicious clients
    if sum(result) > len(result) / 2:
        result = 1 - result

    return result


def agg_pca_agglomer(client_params, pca_d, round, logdir, dataset):
    user_grads = []
    for client in client_params:
        local_parameters = client_params[client]
        if dataset == 'cifar10':
            local_parameters = (local_parameters.reshape(-1, 2))[:, 0]
        user_grads = local_parameters[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_parameters[None, :]), 0)

    param = user_grads.cpu().numpy()
    param, _ = PCA_skl(param, pca_d)

    # if round == 30:
    #     # param_matrix = []
    #     # for client in clients_in_comm:
    #     #     param_matrix.append(param_to_list(client_params[client]))
    #
    #     mkdirs(logdir + '/params/')
    #     euclidean_path = logdir + '/params/' + '/params_round' + str(round) + '.csv'
    #     pd.DataFrame(data=param).to_csv(euclidean_path, index=False, header=False)

    # bc = Birch(n_clusters=2).fit(grad)
    # result = bc.labels_

    agglomer = AgglomerativeClustering(n_clusters=2, linkage='ward', compute_distances=True).fit(param)

    linkage_matrix = get_linkage_matrix(agglomer)
    # 绘制层次聚类树
    if round % 10 == 0:
        fig = plt.figure(figsize=(16, 12))
        dendrogram(linkage_matrix, distance_sort=True, count_sort=True)
        plt.show()
        mkdirs(logdir + '/tree')
        fig.savefig(logdir + '/tree/agglom_' + str(round) + '.png')

    # 构建二叉树
    tree = Building_tree(linkage_matrix, len(agglomer.labels_))
    # 去除异常值得到聚类结果
    min_cluster_size = 3
    benign, malicious, outlier_all = Removing_outliers(tree, min_cluster_size)
    # 得到标签（0，1，-1）

    labels = np.ones(len(agglomer.labels_))
    for value in benign:
        labels[int(value)] = 0

    return labels

    # clients0 = []
    # clients1 = []
    #
    # for id, value in enumerate(result):
    #     if value == 0:
    #         clients0.append('client{}'.format(id))
    #     else:
    #         clients1.append('client{}'.format(id))
    #
    # if len(clients0) >= len(clients1):
    #     benign_clients = clients0
    #     malicious_clients = clients1
    # else:
    #     benign_clients = clients1
    #     malicious_clients = clients0
    #
    # benign_client_params = {}
    # for client in client_params:
    #     if client in benign_clients:
    #         benign_client_params[client] = client_params[client]
    #
    # global_parameters = agg_average(benign_client_params)
    #
    #
    # return global_parameters, malicious_clients


def agg_pca_hdbscan(client_params, pca_d, r):
    user_grads = []
    for client in client_params:
        local_parameters = client_params[client]
        user_grads = local_parameters[None, :] if len(user_grads) == 0 else torch.cat(
            (user_grads, local_parameters[None, :]), 0)

    grad = user_grads.cpu().numpy()
    grad, _ = PCA_skl(grad, pca_d)

    clusterer = HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
    clusterer.fit(grad)
    labels = clusterer.labels_

    # if r % 2 == 0:
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure(figsize=(12, 9))
    #
    #     # clusterer.minimum_spanning_tree_.plot(edge_cmap='viridis',
    #     #                                       edge_alpha=0.6,
    #     #                                       node_size=80,
    #     #                                       edge_linewidth=2)
    #     clusterer.condensed_tree_.plot(label_clusters=labels)
    #
    #     plt.show()
    #     fig.savefig("./tree" + '/hdbscan_' + str(r) + '.png')

    print(labels)
    result = []
    for value in labels:
        if value == Counter(labels).most_common(1)[0][0]:
            result.append(0)
        else:
            result.append(1)
    # 1 is the label of malicious clients

    # if sum(result) > len(result) / 2:
    #     result = 1 - result

    return np.array(result), grad
