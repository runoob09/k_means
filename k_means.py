import time

import numpy
import numpy as np
from math import sqrt


# 选择最近的点，加入到其中
def choose_near_point(X, centers):
    # 计算两点之间的欧式距离
    def distance(X1, X2):
        result = 0
        for (x1, x2) in zip(X1, X2):
            result += (x1 - x2) ** 2
        return sqrt(result)

    # 计算与各中心点距离
    d = [distance(X, i) for i in centers]
    idx = d.index(np.min(d))
    return idx, X


# 求向量的均值
def mean_v(X):
    shape = np.array(X).shape
    mean = [0] * shape[1]
    for x in X:
        for i in range(shape[1]):
            mean[i] += x[i]
    for i in range(shape[1]):
        mean[i] /= shape[0]
    return np.array(mean)


# 计算最新中心点
def update_centroids(group, n_clusters, shape):
    centroids = []
    for i in range(n_clusters):
        if len(group[i]) == 0:
            centroids.append(np.random.rand(shape[1]))
        else:
            mean = mean_v(group[i])
            centroids.append(mean)
    return np.array(centroids)


class KMeans:
    def __init__(self, n_clusters, max_iterations=100):
        # 簇数
        self.n_clusters = n_clusters
        # 最大迭代次数
        self.max_iterations = max_iterations
        # 类簇中心
        self.centroids = None
        # 聚类结果
        self.groups = {}

    def train(self, X):
        # 初始化一个集合用于存储对应的点，用于后续更新点坐标
        def init_group():
            d = {}
            for i in range(self.n_clusters):
                d[i] = []
            return d

        shape = np.array(X).shape
        # 随机生成列簇中心
        self.centroids = np.random.rand(self.n_clusters, shape[1])
        # 开始迭代循环
        for _ in range(self.max_iterations):
            # 按照最近点进行分组
            group = init_group()
            for x in X:
                idx, point = choose_near_point(x, self.centroids)
                group[idx].append(point)
            # 按照分组计算最新中心点值
            new = update_centroids(group, self.n_clusters, shape)
            self.groups = group
            if np.all(np.equal(new, self.centroids)):
                break
            self.centroids = new

    # 预测函数，即选出最近的点
    def predict(self, X):
        return choose_near_point(X, self.centroids)
