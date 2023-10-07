import numpy as np
from numba import cuda


@cuda.jit
def print_thread_idx():
    thread_idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    print(cuda.blockIdx.x)
    print(cuda.blockDim.x)
    print(cuda.threadIdx.x)


print_thread_idx[2, 2]()
