#pragma once // 只编译一次
void softmaxParallel(float *host_destination, float *host_src, int axis, int nDim, int *shape);
void softmaxCnnl(float *host_destination, float *host_src, int nDim, int axis, int *shape);




