from k_means import KMeans as K1
from k_means_mutiprocess import KMeans as K2
from k_means_numba import KMeans as K3
from data import generate_cluster_data
import timeit
from matplotlib import pyplot as plt
import numpy as np


def plot_scatter(model):
    print(model.centroids)
    # 绘制图形
    n_clusters = len(model.groups.keys())
    colors = np.random.rand(n_clusters, 3)
    for i in range(n_clusters):
        x_cluster = np.array(model.groups.get(i))  # 将x_cluster转换为NumPy数组
        if len(x_cluster) == 0:
            continue
        plt.scatter(x_cluster[:, 0], x_cluster[:, 1], color=colors[i])
        plt.scatter(model.centroids[i][0], model.centroids[i][1], color='red')
    plt.show()


if __name__ == '__main__':
    x = generate_cluster_data(5, 10, 5)
    # t1 = timeit.timeit(lambda: K1.train(x), number=3)
    # t2 = timeit.timeit(lambda: K2(n_clusters=5, n_processes=8).train(x), number=3)
    t3 = timeit.timeit(lambda: K3(n_clusters=5,max_iterations=10).train(x))
    # print(t1)
    # 绘制图形
    # print(t2)
    print(t3)
