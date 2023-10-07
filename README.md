# k_means算法的实现
***
## 概述
本次项目使用python实现了k_means算法，从以下几个方面来实现该算法：
* 实现基本算法的功能将其封装为类，向外部提供基本的`train`和`predict`方法
  * 使用串行编程实现上述算法
  * 使用多进程编程实现上述算法
  * 使用numba中的cuda，通过gpu加速运算
* 使用相同的数据，对比不同形式下的算法效率
***
## 项目结构
项目中各文件的用途如下
* data.py：在该文件内部实现了数据生成函数。主要负责数据的生成。
* k_means.py: 使用串行实现的k_means算法
* k_means_multiprocess.py: 使用多进程加速实现的k_means算法
* k_means_numba.py: 使用numba的cuda加速计算的k_means算法