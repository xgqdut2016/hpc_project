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

template <int Br, int Bc>
__device__ float
matmulQK(const float *__restrict inputQ, const float *__restrict inputK,
         float *Qds, float *Kds, int N, int d, int width, int indQ, int indK)
{
    float sum_qk = 0.0f;
    for (int ph = 0; ph < width; ph++)
    {
        if (indQ < N && threadIdx.x + ph * Bc < d)
        {
            Qds[threadIdx.y * Bc + threadIdx.x] =
                inputQ[indQ * d + threadIdx.x + ph * Bc];
        }
        else
        {
            Qds[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        if (threadIdx.y < Bc)
        {
            Kds[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        if (threadIdx.y < Bc)
        {
            if (indK < N && threadIdx.y + ph * Bc < d)
            {
                Kds[threadIdx.y * Bc + threadIdx.x] =
                    inputK[indK * d + threadIdx.y + ph * Bc];
            }
        }

        __syncthreads();
        for (int index = 0; index < Bc; index++)
        {
            sum_qk = std::fma(Qds[threadIdx.y * Bc + index],
                              Kds[index * Bc + threadIdx.x], sum_qk);
        }
        __syncthreads();
    }
    return sum_qk;
}
template <int Br, int Bc>
__device__ float matmulSV(float *sumQK, const float *__restrict inputV,
                          float *Vds, int N, int d, int j, int indQ, int indK,
                          int indV, float tmp_qk, float newMax)
{
    float sumSV = 0.0f;
    if (threadIdx.y < Bc)
    {
        if (threadIdx.y + j * Bc < N && indV < d)
        {
            Vds[threadIdx.x * Bc + threadIdx.y] =
                inputV[(threadIdx.y + j * Bc) * d + indV];
        }
        else
        {
            Vds[threadIdx.x * Bc + threadIdx.y] = 0.0f;
        }
    }
    if (indQ < N && indK < N)
    {
        sumQK[threadIdx.y * Bc + threadIdx.x] = __expf(tmp_qk - newMax);
    }
    else
    {
        sumQK[threadIdx.y * Bc + threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int phc = 0; phc < Bc; phc++)
    {
        sumSV = std::fma(sumQK[threadIdx.y * Bc + phc],
                         Vds[threadIdx.x * Bc + phc], sumSV);
    }
    return sumSV;
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
template <int Br, int Bc>
__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 float *__restrict output)
{

    int Tc = (N + Bc - 1) / Bc;

    __shared__ float sumQK[Br * Bc];
    float sumSV;
    __shared__ float block_max[Br];
    __shared__ float block_sum[Br];
    __shared__ float Vds[Bc * Bc];
    __shared__ float Qds[Br * Bc];
    __shared__ float Kds[Bc * Bc];
    int indV = threadIdx.x + blockIdx.x * blockDim.x;
    int indQ = threadIdx.y + blockIdx.y * blockDim.y;
    float newMax;
    float oldMax;
    float newSum;
    newMax = -__FLT_MAX__;
    oldMax = -__FLT_MAX__;
    newSum = 0.0f;

    float out = 0.0f;
    for (int j = 0; j < Tc; j++)
    {

        int indK = threadIdx.x + j * Bc;
        float sum_qk = 0.0f;
        float tmp_qk = 0.0f;
        sum_qk = matmulQK<Br, Bc>(inputQ, inputK, Qds, Kds, N, d, gridDim.x,
                                  indQ, indK);
        if (indQ < N && indK < N)
        {
            tmp_qk = sum_qk;
        }
        else
        {
            sum_qk = -__FLT_MAX__;
            tmp_qk = 0.0f;
        }
        __syncthreads();
        // softmax reduce
        sum_qk = WarpAllReduce<MaxOp, float, Bc>(sum_qk);
        if (threadIdx.x == 0)
        {
            block_max[threadIdx.y] = sum_qk;
        }
        __syncthreads();
        float localMax = block_max[threadIdx.y];
        //--------------------
        float sum_s = 0.0f;
        if (indQ < N && indK < N)
        {
            sum_s = __expf(tmp_qk - localMax);
        }
        sum_s = WarpAllReduce<SumOp, float, Bc>(sum_s);
        if (threadIdx.x == 0)
        {
            block_sum[threadIdx.y] = sum_s;
        }
        __syncthreads();
        float localSum = block_sum[threadIdx.y];
        if (newMax > localMax)
        {
            newSum = std::fma(localSum, __expf(localMax - newMax), newSum);
            // newSum = newSum + localSum * __expf(localMax - newMax);
        }
        else
        {
            newSum = std::fma(newSum, __expf(newMax - localMax), localSum);
            // newSum = localSum + newSum * __expf(newMax - localMax);
            newMax = localMax;
        }
        sumSV = matmulSV<Br, Bc>(sumQK, inputV, Vds, N, d, j, indQ, indK, indV,
                                 tmp_qk, newMax);
        out = std::fma(__expf(oldMax - newMax), out, sumSV);
        // out = __expf(oldMax - newMax) * out + sumSV;
        oldMax = newMax;

        //__syncthreads();
    }
    if (indQ < N && indV < d)
    {
        output[indQ * d + indV] = out * __fdividef(1.0F, newSum);
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

    int Br = 32;
    int Bc = 32; // Br>=Bc

    int num_block_x = (d + Bc - 1) / Bc;
    int num_block_y = (N + Br - 1) / Br;
    dim3 block_dim(Bc, Br, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    _attentionKernel<32, 32>
        <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    cudaDeviceSynchronize();
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
    int N = 4;
    int d = 3;

    int size = N * d;
    float a[8];
    float *cpu_Q, *cpu_K, *cpu_V, *cpu_output;
    cpu_Q = (float *)malloc(size * sizeof(float));
    cpu_K = (float *)malloc(size * sizeof(float));
    cpu_V = (float *)malloc(size * sizeof(float));
    cpu_output = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        cpu_Q[i] = i;
        cpu_K[i] = i % 4;
        cpu_V[i] = i % 4;
        // printf("Q:%.4f\n",cpu_Q[i]);
    }
    (float4 &)a[0] = (float4 &)cpu_Q[0];
    (float4 &)a[4] = (float4 &)cpu_Q[4];
    printf("[%.2f,%.2f,%.2f,%.2f]\n", a[0], a[1], a[2], a[3]);
    printf("[%.2f,%.2f,%.2f,%.2f]\n", a[4], a[5], a[6], a[7]);
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



