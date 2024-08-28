这里分别展示了不同算法对矩阵乘法性能的影响，根据实验数据，当M-K-N扩大到一定程度情况，手写kernel可以达到cublas性能的95%

matmul_tensor_core.cu：这个代码展示一般形式的tensor core计算C=AB

tensorcore.cu：这个代码详细介绍了tensor core计算C=AB的内部机理

transposeTensor.cu：这个代码展示tensor core借助额外的内存如何计算C=AB.T

tranmat.cu：这个代码展示tensor core如何计算C=AB.T