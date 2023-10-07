import k_means
from k_means_numba import *
from data import *

x = generate_cluster_data(5, 5, 5)
print(mean_v(x))
print(k_means.mean_v(x))