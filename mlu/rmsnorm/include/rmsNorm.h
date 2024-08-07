#pragma once // 只编译一次
void rmsNormParallel(float *host_destination, float *host_src, float *host_weight, int num, int othersize, int dimsize, float eps);
void rmsnormCnnl(float *host_destination, float *host_src, float *host_weight, int nDim, int *shape, float eps);


