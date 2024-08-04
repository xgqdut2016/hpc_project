#include <iostream>
#include <math.h>
#include "softmax.h"
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

int main()
{
    int nDim = 2;
    int shape[nDim] = {1024 * 1024, 1024};
    //int shape[nDim] = {1024 , 1, 1024, 1023};
    //int axis = nDim - 1;
    int axis = 0;
    int dimsize = shape[axis];
    int num = 1;
    for (int s = 0; s < nDim; s++)
    {
        num *= shape[s];
    }
    

    float *host_destination = (float *)malloc(num * sizeof(float));
    float *tmp_destination = (float *)malloc(num * sizeof(float));
    float *host_src = (float *)malloc(num * sizeof(float));
    

    for (int i = 0; i < num; i++)
    {
        host_src[i] = (i % 4) * 1e-1;
    }
    
    
    softmaxCnnl(tmp_destination, host_src, nDim, axis, shape);
    softmaxParallel(host_destination, host_src, axis, nDim, shape);
    
    float err = 0;
    for (int i = 0; i < num; i++)
    {
        err = fmax(err, fabs(tmp_destination[i] - host_destination[i]));
        if (err > 1e-3)
        {
            printf("%d = [%d * dimsize +  %d], error:%.4e, cnnl:%.4e, bangC:%.4e\n", i, i / dimsize, i % dimsize, err, tmp_destination[i], host_destination[i]);
            break;
        }
    }
    printf("%.4e,%.4e\n",tmp_destination[0], host_destination[0]);
    free(host_destination);
    free(tmp_destination);
    free(host_src);
    
    return 0;
}


