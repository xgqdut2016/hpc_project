#include <iostream>
#include <math.h>
#include "rmsNorm.h"
#include <random>

class RandomGenerator
{
private:
    double l, r;
    std::mt19937 e;
    std::uniform_int_distribution<int> di;
    std::uniform_real_distribution<float> dr;

public:
    RandomGenerator(double l = 0, double r = 1, unsigned int seed = 0)
        : l(l), r(r), e(seed), di(l, r), dr(l, r) {}
    virtual ~RandomGenerator() {}
    void fill(uint32_t *data, size_t size)
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] = di(e);
        }
    }
    void fill(float *data, size_t size)
    {
        for (size_t i = 0; i < size; i++)
        {
            data[i] = dr(e);
        }
    }
};
template <typename T>
void rmsNormHost(T *host_destination, T *host_src, T *host_weight, int othersize, int dimsize, float eps)
{
    T globalSum;
    for (int i = 0; i < othersize; i++)
    {
        globalSum = 0.0;
        for (int s = 0; s < dimsize; s++)
        {
            globalSum += host_src[i * dimsize + s] * host_src[i * dimsize + s];
        }
        globalSum += eps;
        T globalSumInv = 1.0 / sqrt(globalSum / dimsize);
        if (i == 4)
        {
            printf("%.4e, %.4e, %.4e\n", globalSumInv, host_src[i * dimsize + 1], host_weight[1]);
        }
        for (int s = 0; s < dimsize; s++)
        {
            host_destination[i * dimsize + s] = host_src[i * dimsize + s] * host_weight[s] * globalSumInv;
        }
    }
}
int main()
{
    int nDim = 2;
    int shape[2] = {1024 * 1024, 1024};
    //int shape[4] = {1024 , 1, 1024, 1024};
    float eps = 1e-5;
    int dimsize = shape[nDim - 1];
    int othersize = 1;
    for (int s = 0; s < nDim - 1; s++)
    {
        othersize *= shape[s];
    }
    int num = othersize * dimsize;

    float *host_destination = (float *)malloc(num * sizeof(float));
    float *tmp_destination = (float *)malloc(num * sizeof(float));
    float *host_src = (float *)malloc(num * sizeof(float));
    float *host_weight = (float *)malloc(dimsize * sizeof(float));

    for (int i = 0; i < num; i++)
    {
        host_src[i] = (i % 4) * 1e-1;
    }
    for (int i = 0; i < dimsize; i++)
    {
        host_weight[i] = i % 3;
    }
    
    rmsnormCnnl(tmp_destination, host_src, host_weight, nDim, shape, eps);
    rmsNormParallel(host_destination, host_src, host_weight, num, othersize, dimsize, eps);
    //rmsNormHost(tmp_destination, host_src, host_weight, othersize, dimsize, eps);
    float err = 0;
    for (int i = 0; i < num; i++)
    {
        err = fmax(err, fabs(tmp_destination[i] - host_destination[i]));
        if (err > 1e-3)
        {
            printf("[%d * dimsize +  %d]error:%.4e, serial:%.4e, parallel:%.4e\n", i / dimsize, i % dimsize, err, tmp_destination[i], host_destination[i]);
            break;
        }
    }

    free(host_destination);
    free(tmp_destination);
    free(host_src);
    free(host_weight);

    return 0;
}



