CUDA代码编译命令为：

nvcc he.cu -o he

## 1Dsoftmax
一维向量softmax代码，参考链接https://blog.csdn.net/forrestguang/article/details/132654913

## attention
flash attention代码，参考链接https://blog.csdn.net/forrestguang/article/details/140667762

## base
基础代码介绍

## highSoftmax
高维向量softmax代码，参考链接https://blog.csdn.net/forrestguang/article/details/134128794

## matrix
cuda core和tensor core实现matmul

## jacobi
九点差分格式的jacobi迭代求解，这个文件夹直接使用make命令就可以自动编译生成可执行文件，参考链接https://blog.csdn.net/forrestguang/article/details/128266791

## MG
多重网格算法求解数值PDE，参考链接https://blog.csdn.net/forrestguang/article/details/131144688

## sparse
稀疏矩阵结合jacobi迭代求解线性系统，参考链接https://blog.csdn.net/forrestguang/article/details/128291289

代码运行顺序如下所示：

首先make create，这个命令会生成稀疏矩阵，右端项，以及精确解

然后使用make，这个命令会根据算法生成可执行文件jacobi

然后使用./jacobi计算

最后使用make clean，此时可以把之前产生的中间文件全部清除

## minres
稀疏矩阵结合minres求解线性系统，参考链接https://blog.csdn.net/forrestguang/article/details/128291289

代码运行顺序如下所示：

首先make create，这个命令会生成稀疏矩阵，右端项，以及精确解

然后使用make，这个命令会根据算法生成可执行文件minres

然后使用./minres计算

最后使用make clean，此时可以把之前产生的中间文件全部清除