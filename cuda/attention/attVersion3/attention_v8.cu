#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>
void getThreadNum();

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

const int Rq = 8;
const int Rv = 4;
const int BLOCK_DIM_y = 16;
const int BLOCK_DIM_x = 16;

const int Br = BLOCK_DIM_y * Rq;
const int Bc = BLOCK_DIM_x * Rv;
const int Bk = 4;
const int Bd = 8;

template <int BLOCK_DIM_y, int BLOCK_DIM_x, int Rq>
__device__ void matmulRQK(const float *__restrict inputQ,
                          const float *__restrict inputK, float *shareQK,
                          float *shareVK, int N, int d, int width, int indQ,
                          int indK, float *val)
{
    for (int index = 0; index < d; index++)
    {
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            for (int index_k = 0; index_k < Bk; index_k++)
            {
                val[index_q * Bk + index_k] +=
                    inputQ[(indQ + index_q) * d + index] *
                    inputK[(indK + index_k) * d + index];
            }
        }
    }
}
template <int BLOCK_DIM_y, int BLOCK_DIM_x, int Rq, int Rv>
__device__ void matmulSV(float *shareQK, const float *__restrict inputV,
                         float *shareVK, int N, int d, int j, int indQ,
                         int indK, int indV, float *val, float *newMax,
                         float *sumSV)
{

    for (int index_q = 0; index_q < Rq; index_q++)
    {
        for (int index_k = 0; index_k < Bk; index_k++)
        {
            if (indQ + index_q < N && indK + index_k < N)
            {
                shareQK[(threadIdx.y * Rq + index_q) * (BLOCK_DIM_x * Bk) +
                        threadIdx.x * Bk + index_k] =
                    __expf(val[index_q * Bk + index_k] - newMax[index_q]);
            }
            else
            {
                shareQK[(threadIdx.y * Rq + index_q) * (BLOCK_DIM_x * Bk) +
                        threadIdx.x * Bk + index_k] = 0.0f;
            }
        }
    }
    __syncthreads();
    float regLeft;

    for (int phc = 0; phc < BLOCK_DIM_x * Bk; phc++)
    {
        if (phc + j * BLOCK_DIM_x * Bk < N)
        {
            for (int index_q = 0; index_q < Rq; index_q++)
            {
                regLeft =
                    shareQK[(threadIdx.y * Rq + index_q) * (BLOCK_DIM_x * Bk) +
                            phc];

                for (int index_v = 0; index_v < Rv; index_v++)
                {
                    sumSV[index_q * Rv + index_v] +=
                        regLeft * inputV[(phc + j * BLOCK_DIM_x * Bk) * d +
                                         indV + index_v];
                }
            }
        }
    }
}
template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return a + b;
    }
};

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return max(a, b);
    }
};
template <template <typename> class ReductionOp, typename T,
          int thread_group_width = warpSize>
__inline__ __device__ T WarpAllReduce(T val)
{
    for (int mask = thread_group_width / 2; mask > 0; mask >>= 1)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }

    return val;
}

template <int BLOCK_DIM_y, int BLOCK_DIM_x, int Rq, int Rv>
__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 float *__restrict output)
{

    __shared__ float shareQK[Br * BLOCK_DIM_x * Bk];
    __shared__ float shareVK[BLOCK_DIM_x * Bk * Bc];

    float sumSV[Rq * Rv] = {0.0f};
    float newMax[Rq];
    float oldMax[Rq];
    float newSum[Rq] = {0.0f};

    float val[Rq * Bk];

    int indV = Rv * (threadIdx.x + blockIdx.x * blockDim.x);
    int indQ = Rq * (threadIdx.y + blockIdx.y * blockDim.y);

    for (int index_q = 0; index_q < Rq; index_q++)
    {
        newMax[index_q] = -__FLT_MAX__;
        oldMax[index_q] = -__FLT_MAX__;
    }

    int Tc = (N + BLOCK_DIM_x * Bk - 1) / (BLOCK_DIM_x * Bk);

    int width = (d + Bd - 1) / Bd;
    for (int j = 0; j < Tc; j++)
    {

        int indK = Bk * (threadIdx.x + j * BLOCK_DIM_x);
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            for (int index_k = 0; index_k < Bk; index_k++)
            {
                val[index_q * Bk + index_k] = 0.0f;
            }
        }

        matmulRQK<BLOCK_DIM_y, BLOCK_DIM_x, Rq>(
            inputQ, inputK, shareQK, shareVK, N, d, width, indQ, indK, val);
        __syncthreads();
        for (int index_q = 0; index_q < Rq; index_q++)
        {
            float tmpReduceMax = -__FLT_MAX__;
            for (int index_k = 0; index_k < Bk; index_k++)
            {
                if (indQ + index_q < N && indK + index_k < N)
                {
                    tmpReduceMax =
                        max(val[index_q * Bk + index_k], tmpReduceMax);
                }
            }
            __syncthreads();
            tmpReduceMax =
                WarpAllReduce<MaxOp, float, BLOCK_DIM_x>(tmpReduceMax);
            if (threadIdx.x == 0)
            {
                shareQK[threadIdx.y * Rq + index_q] =
                    tmpReduceMax; // block_max[Br] = Max,Br=BLOCK_DIM_y*Rq
            }
            __syncthreads();
            float tmpReduceSum = 0.0f;
            for (int index_k = 0; index_k < Bk; index_k++)
            {
                if (indQ + index_q < N && indK + index_k < N)
                {
                    tmpReduceSum += __expf(val[index_q * Bk + index_k] -
                                           shareQK[threadIdx.y * Rq + index_q]);
                }
            }
            __syncthreads();
            tmpReduceSum =
                WarpAllReduce<SumOp, float, BLOCK_DIM_x>(tmpReduceSum);
            if (threadIdx.x == 0)
            {
                shareQK[threadIdx.y * Rq + index_q + Br] = tmpReduceSum;
            }
            __syncthreads();
            if (newMax[index_q] > shareQK[threadIdx.y * Rq + index_q])
            {
                newSum[index_q] =
                    std::fma(shareQK[threadIdx.y * Rq + index_q + Br],
                             __expf(shareQK[threadIdx.y * Rq + index_q] -
                                    newMax[index_q]),
                             newSum[index_q]);
            }
            else
            {
                newSum[index_q] =
                    std::fma(newSum[index_q],
                             __expf(newMax[index_q] -
                                    shareQK[threadIdx.y * Rq + index_q]),
                             shareQK[threadIdx.y * Rq + index_q + Br]);

                newMax[index_q] = shareQK[threadIdx.y * Rq + index_q];
            }

            // PV
            for (int index_v = 0; index_v < Rv; index_v++)
            {
                sumSV[index_q * Rv + index_v] *=
                    __expf(oldMax[index_q] - newMax[index_q]);
            }
        }

        matmulSV<BLOCK_DIM_y, BLOCK_DIM_x, Rq, Rv>(shareQK, inputV, shareVK, N,
                                                   d, j, indQ, indK, indV, val,
                                                   newMax, sumSV);

        for (int index_q = 0; index_q < Rq; index_q++)
        {
            oldMax[index_q] = newMax[index_q];
        }

        __syncthreads();
    }
    __syncthreads();

    for (int index_q = 0; index_q < Rq; index_q++)
    {
        float inv = __fdividef(1.0F, newSum[index_q]);
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            sumSV[index_q * Rv + index_v] = sumSV[index_q * Rv + index_v] * inv;
        }
    }
    for (int index_q = 0; index_q < Rq; index_q++)
    {
        for (int index_v = 0; index_v < Rv; index_v++)
        {
            if (indQ + index_q < N && indV + index_v < d)
            {
                output[(indQ + index_q) * d + indV + index_v] =
                    sumSV[index_q * Rv + index_v];
            }
        }
    }
}
void attention(float *cpu_Q, float *cpu_K, float *cpu_V, int N, int d, float *cpu_output)
{
    double st, ela;
    st = get_walltime();

    float *inputQ, *inputK, *inputV, *output;
    cudaMalloc((void **)&inputQ, N * d * sizeof(float));
    cudaMalloc((void **)&inputK, N * d * sizeof(float));
    cudaMalloc((void **)&inputV, N * d * sizeof(float));

    cudaMalloc((void **)&output, N * d * sizeof(float));
    cudaMemcpy(inputQ, cpu_Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inputK, cpu_K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inputV, cpu_V, N * d * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int num_block_x = (d + Bc - 1) / Bc;
    int num_block_y = (N + Br - 1) / Br;
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);

    _attentionKernel<BLOCK_DIM_y, BLOCK_DIM_x, Rq, Rv>
        <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(cpu_output, output, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(inputQ);
    cudaFree(inputK);
    cudaFree(inputV);

    cudaFree(output);

    ela = get_walltime() - st;

    printf("BlockReduce,kernel time:%.4f, use time:%.4f\n", ker_time / 1000., ela);
    // getThreadNum();
}
int main()
{
    int N = 1024;
    int d = 128;

    int size = N * d;

    float *cpu_Q, *cpu_K, *cpu_V, *cpu_output;
    cpu_Q = (float *)malloc(size * sizeof(float));
    cpu_K = (float *)malloc(size * sizeof(float));
    cpu_V = (float *)malloc(size * sizeof(float));
    cpu_output = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        cpu_Q[i] = i % 4;
        cpu_K[i] = i % 4;
        cpu_V[i] = i % 4;
        // printf("Q:%.4f\n",cpu_Q[i]);
    }

    attention(cpu_Q, cpu_K, cpu_V, N, d, cpu_output);
    for (int i = 0; i < 10; i++)
    {

        printf("out:%.6e\n", cpu_output[i]);
    }

    free(cpu_Q);
    free(cpu_K);
    free(cpu_V);
    free(cpu_output);

    return 0;
}
void getThreadNum()
{
    int deviceCount;

    cudaGetDeviceCount(&deviceCount); // Returns in *deviceCount the number of devices
    printf("deviceCount: %d\n ", deviceCount);

    if (deviceCount == 0)
    {
        printf("error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    cudaSetDevice(dev); // Sets dev=0 device as the current device for the calling host thread.

    cudaDeviceProp devProps;
    cudaGetDeviceProperties(&devProps, dev);
    printf("name:%s\n", devProps.name);
    printf("totalGlobalMem: %ld\n", devProps.totalGlobalMem);
    printf("regsPerBlock: %d\n", devProps.regsPerBlock);
    printf("warpSize: %d\n", devProps.warpSize);
    printf("memPitch: %ld\n\n", devProps.memPitch);

    printf("一个线程块中可使用的最大共享内存\n");
    printf("devProps.sharedMemPerBlock: %ld Bytes \n\n", devProps.sharedMemPerBlock);

    printf("一个线程块中可包含的最大线程数量\n");
    printf("maxThreadsPerBlock: %d\n", devProps.maxThreadsPerBlock);

    printf("多维线程块数组中每一维可包含的最大线程数量\n");
    printf("maxThreadsDim[0]: %d\n", devProps.maxThreadsDim[0]);
    printf("maxThreadsDim[1]: %d\n", devProps.maxThreadsDim[1]);
    printf("maxThreadsDim[2]: %d\n\n", devProps.maxThreadsDim[2]);

    printf("一个线程格中每一维可包含的最大线程块数量\n");
    printf("maxGridSize[0]: %d\n", devProps.maxGridSize[0]);
    printf("maxGridSize[1]: %d\n", devProps.maxGridSize[1]);
    printf("maxGridSize[2]: %d\n\n", devProps.maxGridSize[2]);

    printf("clockRate: %d\n", devProps.clockRate);
    printf("totalConstMem: %ld\n", devProps.totalConstMem);
    printf("textureAlignment: %ld\n\n", devProps.textureAlignment);

    printf("计算能力：%d.%d\n", devProps.major, devProps.minor);
}


