from k_means import KMeans as K1
from k_means_mutiprocess import KMeans as K2
from k_means_numba import KMeans as K3
from data import generate_cluster_data
import timeit
from matplotlib import pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"

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
    x = generate_cluster_data(20000, 1000, 5)
    # t1 = timeit.timeit(lambda: K1(n_clusters=5,max_iterations=10).train(x), number=1)
    # print("串行耗时：{:.2f}秒".format(t1))
    t2 = timeit.timeit(lambda: K2(n_clusters=5, n_processes=8,max_iterations=10).train(x), number=1)
    print("多进程耗时：{:.2f}秒".format(t2))
    t3 = timeit.timeit(lambda: K3(n_clusters=5,max_iterations=10).train(x),number=1)
    print("GPU加速耗时：{:.2f}秒".format(t3))
