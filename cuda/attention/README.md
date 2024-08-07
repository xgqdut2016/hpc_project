v1,v2,v3,v4参考链接https://blog.csdn.net/forrestguang/article/details/133310063

v5,v6,v7,v8,v9,v10参考链接https://blog.csdn.net/forrestguang/article/details/134487740

v5：Q[i,:]=blockIdx.y,V[:,j]=blockIdx.x × blockDim.x + threadIdx.x

v6：规约加速

v7：Q[i,:]=blockIdx.x × blockDim.x + threadIdx.x,V[:,j]=blockIdx.y

v8：一维线程块

v9：shuffle warp规约

v10：内存复用