import time

import numba
import numpy as np
from math import sqrt
from numba import cuda


@cuda.jit
def euclidean_distance_gpu(vector1, vector2, result):
    i = cuda.grid(1)
    if i < vector1.shape[0]:
        diff = vector1[i] - vector2[i]
        result[i] = diff * diff

def distance(vector1, vector2):
    # 将向量移动到 GPU 内存
    d_vector1 = cuda.to_device(vector1)
    d_vector2 = cuda.to_device(vector2)
    # 创建结果
    result = cuda.device_array_like(vector1)
    # 配置 CUDA 栅格和线程块
    threads_per_block = 2
    blocks_per_grid = (vector1.size + (threads_per_block - 1)) // threads_per_block
    # 调用 CUDA 函数计算欧氏距离的平方
    euclidean_distance_gpu[blocks_per_grid, threads_per_block](d_vector1, d_vector2, result)
    # 将结果移回到主机内存
    result_host = result.copy_to_host()
    # 在主机上计算结果的平方根
    distance = sqrt(result_host.sum())
    return distance


def choose_near_point(X, centers):
    d = np.zeros(len(centers))
    # 创建结果数组
    start = time.time()
    for i in range(len(centers)):
        d[i] = distance(X, centers[i])
    end = time.time()
    print(end-start)
    idx = np.argmin(d)
    return idx, X


@cuda.jit
def compute_mean_gpu(result, matrix):
    idx = cuda.grid(1)
    if idx < matrix.shape[1]:
        sum_val = 0.0
        for i in range(matrix.shape[0]):
            sum_val += matrix[i, idx]
        result[idx] = sum_val / matrix.shape[0]


def mean_v(vectors):
    # 将向量移动到 GPU 内存
    d_vectors = cuda.to_device(vectors)
    # 创建结果数组
    result = cuda.device_array(vectors.shape[1], dtype=np.float32)
    # 配置 CUDA 栅格和线程块
    threads_per_block = 2
    blocks_per_grid = (vectors.shape[0] + (threads_per_block - 1)) // threads_per_block
    # 调用 CUDA 函数计算向量均值
    compute_mean_gpu[blocks_per_grid, threads_per_block](result, d_vectors)
    # 将结果移回到主机内存
    result_host = result.copy_to_host()
    return result_host


def update_centroids(group, n_clusters, shape):
    centroids = np.zeros((n_clusters, shape[1]))
    for i in range(n_clusters):
        if len(group[i]) == 0:
            centroids[i] = np.random.rand(shape[1])
        else:
            mean = mean_v(np.array(group[i]))
            centroids[i] = mean
    return centroids


class KMeans:
    def __init__(self, n_clusters, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.centroids = None
        self.groups = None

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
            if np.all(np.equal(self.centroids, new)):
                break
            self.centroids = new

    def predict(self, X):
        return choose_near_point(X, self.centroids)
