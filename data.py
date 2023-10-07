import numpy as np


def generate_cluster_data(n_features, n_samples,n_clusters):
    centers = np.random.randn(n_clusters, n_features) * 10  # 生成3个聚类中心点，维度为n_features
    X = np.empty((n_samples, n_features))
    for i in range(n_samples):
        center_index = np.random.randint(0, 3)  # 随机选择一个聚类中心
        center = centers[center_index]
        x = np.random.randn(n_features) + center  # 从聚类中心点生成一个样本
        X[i] = x
    return X
