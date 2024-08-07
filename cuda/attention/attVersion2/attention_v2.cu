#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}

__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 int Br, int Bc, int rate,
                                 float *__restrict output)
{
    // 使用shuffle
    // warp规约必须针对threadIdx.x，而且Bc=blockDim.x不能超过32,Bc必须是常数
    //  一个线程块处理Q的Br行，V的Bc列，以及全部的K,blockDim.x=Bc,blockDim.y=Br
    int Tc = (N + Bc - 1) / Bc;     // 遍历矩阵inputK的N行需要的循环次数
    extern __shared__ float sram[]; // 必须要有extern
    float *sumQK = sram;            // 形状为[Bc,Br]，存储的是QK.T的结果
    float *sumSV =
        sram + Bc * Br; // 形状为[Bc,Br]，存储的是softmax(QK.T)V的结果

    int indQ =
        threadIdx.y +
        blockIdx.y *
            blockDim
                .y; // 对应的是当前block需要处理的Q的行索引
                    // 如果ceil(d/Bc)=gridDim.x >
                    // rate，说明d太大，共享内存放不下，此时共享内存就放对应i的前rate×Bc的Q元素
    float *Qds =
        sram +
        Bc * Br * 2; // 如果gridDim.x <= rate,那么正好把Q的Br行元素全部存储
    int sEnd =
        min(rate, gridDim.x); // 后面申请共享内存的时候也是申请Br×sEnd×Bc的空间
    for (int s = 0; s < sEnd; s++)
    {
        if (threadIdx.x + s * Bc < d)
        {
            Qds[threadIdx.y * sEnd * Bc + threadIdx.x + s * Bc] =
                inputQ[indQ * d + threadIdx.x + s * Bc];
        }
        else
        {
            Qds[threadIdx.y * sEnd * Bc + threadIdx.x + s * Bc] = 0.0f;
        }
    }
    __syncthreads();
    int indV = threadIdx.x +
               blockIdx.x * blockDim.x; // 对应的是当前block需要处理的V的列索引

    float *block_max = sram + Bc * Br * 2 + Bc * Br * sEnd;
    float *block_sum = sram + Bc * Br * 2 + Bc * Br * sEnd + Br * Bc;
    float newMax;
    float oldMax;
    float newSum;
    newMax = -__FLT_MAX__;
    oldMax = -__FLT_MAX__;
    newSum = 1.0f;

    float out = 0.0f;
    for (int j = 0; j < Tc; j++)
    {
        sumSV[threadIdx.y * Bc + threadIdx.x] =
            0.0f;                        // 每次循环需要重新初始化为0
        int indK = threadIdx.x + j * Bc; // 通过j循环来遍历K的行索引
        float sum_qk = 0.0f;
        for (int index = 0; index < d; index++)
        {
            if (index < rate * Bc)
            {
                sum_qk += Qds[threadIdx.y * sEnd * Bc + index] *
                          inputK[indK * d + index];
            }
            else
            {
                sum_qk += inputQ[indQ * d + index] * inputK[indK * d + index];
            }
        }
        if (indQ < N && indK < N)
        {
            block_max[threadIdx.y * Bc + threadIdx.x] = sum_qk;
            block_sum[threadIdx.y * Bc + threadIdx.x] = 1.0f;
            sumQK[threadIdx.y * Bc + threadIdx.x] =
                sum_qk; // 存储QK的结果，循环内部不做修改
        }
        else
        {
            block_max[threadIdx.y * Bc + threadIdx.x] = -__FLT_MAX__;
            block_sum[threadIdx.y * Bc + threadIdx.x] = 0.0f;
            sumQK[threadIdx.y * Bc + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int strip = Bc / 2; strip > 0; strip /= 2)
        {
            if (threadIdx.x < strip)
            {
                if (block_max[threadIdx.y * Bc + threadIdx.x] >
                    block_max[threadIdx.y * Bc + threadIdx.x + strip])
                {
                    block_sum[threadIdx.y * Bc + threadIdx.x] =
                        block_sum[threadIdx.y * Bc + threadIdx.x] +
                        block_sum[threadIdx.y * Bc + threadIdx.x + strip] *
                            __expf(block_max[threadIdx.y * Bc + threadIdx.x +
                                             strip] -
                                   block_max[threadIdx.y * Bc + threadIdx.x]);
                }
                else
                {
                    block_sum[threadIdx.y * Bc + threadIdx.x] =
                        block_sum[threadIdx.y * Bc + threadIdx.x + strip] +
                        block_sum[threadIdx.y * Bc + threadIdx.x] *
                            __expf(block_max[threadIdx.y * Bc + threadIdx.x] -
                                   block_max[threadIdx.y * Bc + threadIdx.x +
                                             strip]);
                    block_max[threadIdx.y * Bc + threadIdx.x] =
                        block_max[threadIdx.y * Bc + threadIdx.x + strip];
                }
            }
            __syncthreads();
        }

        if (newMax >
            block_max[threadIdx.y *
                      Bc]) // threadIdx.y=0存储的是对应分块矩阵的局部max
        {                  // 为了获得全局max，需要不断更新newMax和threadIdx.y=0的比较结果
            newSum = newSum + block_sum[threadIdx.y * Bc] *
                                  __expf(block_max[threadIdx.y * Bc] - newMax);
        }
        else
        {
            newSum = block_sum[threadIdx.y * Bc] +
                     newSum * __expf(newMax - block_max[threadIdx.y * Bc]);
            newMax = block_max[threadIdx.y * Bc];
        }

        __syncthreads();

        for (int phc = 0; phc < Bc; phc++) // 这里开始做最后和V的matmul
        {
            if (phc + j * Bc < N) // 注意控制范围
            {
                sumSV[threadIdx.y * Bc + threadIdx.x] +=
                    __expf(sumQK[threadIdx.y * Bc + phc] - newMax) *
                    inputV[(phc + j * Bc) * d + indV];
            }
        }
        out = __expf(oldMax - newMax) * out +
              sumSV[threadIdx.y * Bc + threadIdx.x];
        oldMax = newMax;

        __syncthreads();
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

    int Bc = 32; // 不能超过32
    int Br = 32;
    int num_block_x = (d + Bc - 1) / Bc;
    int num_block_y = (N + Br - 1) / Br;
    dim3 block_dim(Bc, Br, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    int rate = 5; // rate为了防止Br <
                  // d，但是d又太大无法全部把Qi加载到共享内存，设置阈值
    int sEnd = min(rate, num_block_x);
    int share_mem =
        ((4 + sEnd) * Bc * Br) *
        sizeof(
            float); // 由于global函数里面未明确分配内存，此时必须指定共享内存分配大小
    _attentionKernel<<<grid_dim, block_dim, share_mem>>>(
        inputQ, inputK, inputV, N, d, Br, Bc, rate, output);

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
}
int main()
{
    int N = 4;
    int d = 3;

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


