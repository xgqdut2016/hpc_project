#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
const int TM = 8;
const int TN = 8;
const int BLOCK_DIM_x = 16;
const int BLOCK_DIM_y = 16;
const int BM = TM * BLOCK_DIM_x;
const int BN = TN * BLOCK_DIM_y;
const int BK = 8;
#define addSpecial4(destination, alpha, source) \
    destination.x += alpha * source.x;          \
    destination.y += alpha * source.y;          \
    destination.z += alpha * source.z;          \
    destination.w += alpha * source.w;
#define addSpecial(destination, alpha, source) \
    *destination += alpha * source.x;          \
    *(destination + 1) += alpha * source.y;    \
    *(destination + 2) += alpha * source.z;    \
    *(destination + 3) += alpha * source.w;
double
get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int s = 0; s < K; s++)
            {
                tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
void compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    bool tmp = true;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
        if (error > 1e-5)
        {
            tmp = false;
            printf("error:hostC[%d] = %.3f, serialC[%d] = %.3f\n", i, hostC[i], i, serialC[i]);
            break;
        }
    }
    if (tmp)
    {
        printf("GPU output all right\n");
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel3th(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK * 2];
    __shared__ float SB[BK * BN * 2];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    float4 a[1];
    float4 b[1];
    float4 com_a[2];
    float4 com_b[2];
    float4 tmp[16];
    memset(tmp, 0.0f, sizeof(tmp));
    //------------
    int ph = 0;
    (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
    SA[(4 * smem_a_k) * BM + smem_a_m] = a[0].x;
    SA[(4 * smem_a_k + 1) * BM + smem_a_m] = a[0].y;
    SA[(4 * smem_a_k + 2) * BM + smem_a_m] = a[0].z;
    SA[(4 * smem_a_k + 3) * BM + smem_a_m] = a[0].w;
    (float4 &)b[0] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
    (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)b[0];
    for (int id = 0; id < 4; id++)
    {

        if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
        {

            SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
        }
    }
    __syncthreads();

    for (int ph = 1; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];

        (float4 &)b[0] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];

        //-------------
        for (int index_k = 0; index_k < BK; index_k++)
        {
            (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
            (float4 &)com_a[1] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
            (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
            (float4 &)com_b[1] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
            addSpecial4(tmp[0], com_a[0].x, com_b[0]); // index_q = 0, 0<= index_v <= 3
            addSpecial4(tmp[1], com_a[0].x, com_b[1]); // index_q = 0, 4<= index_v <= 7
            addSpecial4(tmp[2], com_a[0].y, com_b[0]); // index_q = 1, 0<= index_v <= 3
            addSpecial4(tmp[3], com_a[0].y, com_b[1]); // index_q = 1, 4<= index_v <= 7
            addSpecial4(tmp[4], com_a[0].z, com_b[0]); // index_q = 2, 0<= index_v <= 3
            addSpecial4(tmp[5], com_a[0].z, com_b[1]); // index_q = 2, 4<= index_v <= 7
            addSpecial4(tmp[6], com_a[0].w, com_b[0]); // index_q = 3, 0<= index_v <= 3
            addSpecial4(tmp[7], com_a[0].w, com_b[1]); // index_q = 3, 4<= index_v <= 7

            addSpecial4(tmp[8], com_a[1].x, com_b[0]);  // index_q = 4, 0<= index_v <= 3
            addSpecial4(tmp[9], com_a[1].x, com_b[1]);  // index_q = 4, 4<= index_v <= 7
            addSpecial4(tmp[10], com_a[1].y, com_b[0]); // index_q =5, 0<= index_v <= 3
            addSpecial4(tmp[11], com_a[1].y, com_b[1]); // index_q = 5, 4<= index_v <= 7
            addSpecial4(tmp[12], com_a[1].z, com_b[0]); // index_q = 6, 0<= index_v <= 3
            addSpecial4(tmp[13], com_a[1].z, com_b[1]); // index_q = 6, 4<= index_v <= 7
            addSpecial4(tmp[14], com_a[1].w, com_b[0]); // index_q =7, 0<= index_v <= 3
            addSpecial4(tmp[15], com_a[1].w, com_b[1]); // index_q = 7, 4<= index_v <= 7
        }
        SA[(4 * smem_a_k) * BM + smem_a_m + ph % 2 * BM * BK] = a[0].x;
        SA[(4 * smem_a_k + 1) * BM + smem_a_m + ph % 2 * BM * BK] = a[0].y;
        SA[(4 * smem_a_k + 2) * BM + smem_a_m + ph % 2 * BM * BK] = a[0].z;
        SA[(4 * smem_a_k + 3) * BM + smem_a_m + ph % 2 * BM * BK] = a[0].w;
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n + ph % 2 * BN * BK] = (float4 &)b[0];
        __syncthreads();
    }
    //--------------
    ph = width;
    for (int index_k = 0; index_k < BK; index_k++)
    {
        (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
        (float4 &)com_a[1] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
        (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
        (float4 &)com_b[1] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
        addSpecial4(tmp[0], com_a[0].x, com_b[0]); // index_q = 0, 0<= index_v <= 3
        addSpecial4(tmp[1], com_a[0].x, com_b[1]); // index_q = 0, 4<= index_v <= 7
        addSpecial4(tmp[2], com_a[0].y, com_b[0]); // index_q = 1, 0<= index_v <= 3
        addSpecial4(tmp[3], com_a[0].y, com_b[1]); // index_q = 1, 4<= index_v <= 7
        addSpecial4(tmp[4], com_a[0].z, com_b[0]); // index_q = 2, 0<= index_v <= 3
        addSpecial4(tmp[5], com_a[0].z, com_b[1]); // index_q = 2, 4<= index_v <= 7
        addSpecial4(tmp[6], com_a[0].w, com_b[0]); // index_q = 3, 0<= index_v <= 3
        addSpecial4(tmp[7], com_a[0].w, com_b[1]); // index_q = 3, 4<= index_v <= 7

        addSpecial4(tmp[8], com_a[1].x, com_b[0]);  // index_q = 4, 0<= index_v <= 3
        addSpecial4(tmp[9], com_a[1].x, com_b[1]);  // index_q = 4, 4<= index_v <= 7
        addSpecial4(tmp[10], com_a[1].y, com_b[0]); // index_q =5, 0<= index_v <= 3
        addSpecial4(tmp[11], com_a[1].y, com_b[1]); // index_q = 5, 4<= index_v <= 7
        addSpecial4(tmp[12], com_a[1].z, com_b[0]); // index_q = 6, 0<= index_v <= 3
        addSpecial4(tmp[13], com_a[1].z, com_b[1]); // index_q = 6, 4<= index_v <= 7
        addSpecial4(tmp[14], com_a[1].w, com_b[0]); // index_q =7, 0<= index_v <= 3
        addSpecial4(tmp[15], com_a[1].w, com_b[1]); // index_q = 7, 4<= index_v <= 7
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        (float4 &)dC[(indA + threadIdx.x * TM + index_q) * N + indB + threadIdx.y * TN] = (float4 &)tmp[2 * index_q];
        (float4 &)dC[(indA + threadIdx.x * TM + index_q) * N + indB + threadIdx.y * TN + 4] = (float4 &)tmp[2 * index_q + 1];
    }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixKernel4th(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float SA[BM * BK * 2];
    __shared__ float SB[BK * BN * 2];
    int indA = TM * (blockIdx.x * blockDim.x);
    int indB = TN * (blockIdx.y * blockDim.y);
    int width = (K + BK - 1) / BK;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int smem_a_m = tid / 2;
    int smem_a_k = tid % 2;
    int smem_b_k = tid / 32;
    int smem_b_n = tid % 32;
    float4 a[1];
    float4 b[1];
    float4 com_a[2];
    float4 com_b[2];
    float tmp[64];
    memset(tmp, 0.0f, sizeof(tmp));
    //------------
    int ph = 0;
    (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];
    SA[(4 * smem_a_k) * BM + smem_a_m] = a[0].x;
    SA[(4 * smem_a_k + 1) * BM + smem_a_m] = a[0].y;
    SA[(4 * smem_a_k + 2) * BM + smem_a_m] = a[0].z;
    SA[(4 * smem_a_k + 3) * BM + smem_a_m] = a[0].w;
    (float4 &)b[0] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];
    (float4 &)SB[smem_b_k * BN + 4 * smem_b_n] = (float4 &)b[0];
    for (int id = 0; id < 4; id++)
    {

        if (indB + 4 * smem_b_n + id >= N || smem_b_k + ph * BK >= K)
        {

            SB[smem_b_k * BN + 4 * smem_b_n + id] = 0.0f;
        }
    }
    __syncthreads();

    for (int ph = 1; ph < width; ph++)
    {
        (float4 &)a[0] = (float4 &)dA[(indA + smem_a_m) * K + 4 * smem_a_k + ph * BK];

        (float4 &)b[0] = (float4 &)dB[(smem_b_k + ph * BK) * N + indB + 4 * smem_b_n];

        //-------------
        for (int index_k = 0; index_k < BK; index_k++)
        {
            (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
            (float4 &)com_a[1] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
            (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
            (float4 &)com_b[1] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
            addSpecial(&tmp[0], com_a[0].x, com_b[0]);          // index_q = 0, 0<= index_v <= 3
            addSpecial(&tmp[4], com_a[0].x, com_b[1]);          // index_q = 0, 4<= index_v <= 7
            addSpecial(&tmp[TN], com_a[0].y, com_b[0]);         // index_q = 1, 0<= index_v <= 3
            addSpecial(&tmp[TN + 4], com_a[0].y, com_b[1]);     // index_q = 1, 4<= index_v <= 7
            addSpecial(&tmp[2 * TN], com_a[0].z, com_b[0]);     // index_q = 2, 0<= index_v <= 3
            addSpecial(&tmp[2 * TN + 4], com_a[0].z, com_b[1]); // index_q = 2, 4<= index_v <= 7
            addSpecial(&tmp[3 * TN], com_a[0].w, com_b[0]);     // index_q = 3, 0<= index_v <= 3
            addSpecial(&tmp[3 * TN + 4], com_a[0].w, com_b[1]); // index_q = 3, 4<= index_v <= 7

            addSpecial(&tmp[4 * TN], com_a[1].x, com_b[0]);     // index_q = 4, 0<= index_v <= 3
            addSpecial(&tmp[4 * TN + 4], com_a[1].x, com_b[1]); // index_q = 4, 4<= index_v <= 7
            addSpecial(&tmp[5 * TN], com_a[1].y, com_b[0]);     // index_q =5, 0<= index_v <= 3
            addSpecial(&tmp[5 * TN + 4], com_a[1].y, com_b[1]); // index_q = 5, 4<= index_v <= 7
            addSpecial(&tmp[6 * TN], com_a[1].z, com_b[0]);     // index_q = 6, 0<= index_v <= 3
            addSpecial(&tmp[6 * TN + 4], com_a[1].z, com_b[1]); // index_q = 6, 4<= index_v <= 7
            addSpecial(&tmp[7 * TN], com_a[1].w, com_b[0]);     // index_q =7, 0<= index_v <= 3
            addSpecial(&tmp[7 * TN + 4], com_a[1].w, com_b[1]); // index_q = 7, 4<= index_v <= 7
        }
        SA[(4 * smem_a_k) * BM + smem_a_m + ph % 2 * BM * BK] = a[0].x;
        SA[(4 * smem_a_k + 1) * BM + smem_a_m + ph % 2 * BM * BK] = a[0].y;
        SA[(4 * smem_a_k + 2) * BM + smem_a_m + ph % 2 * BM * BK] = a[0].z;
        SA[(4 * smem_a_k + 3) * BM + smem_a_m + ph % 2 * BM * BK] = a[0].w;
        (float4 &)SB[smem_b_k * BN + 4 * smem_b_n + ph % 2 * BN * BK] = (float4 &)b[0];
        __syncthreads();
    }
    //--------------
    ph = width;
    for (int index_k = 0; index_k < BK; index_k++)
    {
        (float4 &)com_a[0] = (float4 &)SA[index_k * BM + threadIdx.x * TM + (ph - 1) % 2 * BM * BK];
        (float4 &)com_a[1] = (float4 &)SA[index_k * BM + threadIdx.x * TM + 4 + (ph - 1) % 2 * BM * BK];
        (float4 &)com_b[0] = (float4 &)SB[index_k * BN + threadIdx.y * TN + (ph - 1) % 2 * BN * BK];
        (float4 &)com_b[1] = (float4 &)SB[index_k * BN + threadIdx.y * TN + 4 + (ph - 1) % 2 * BN * BK];
        addSpecial(&tmp[0], com_a[0].x, com_b[0]);          // index_q = 0, 0<= index_v <= 3
        addSpecial(&tmp[4], com_a[0].x, com_b[1]);          // index_q = 0, 4<= index_v <= 7
        addSpecial(&tmp[TN], com_a[0].y, com_b[0]);         // index_q = 1, 0<= index_v <= 3
        addSpecial(&tmp[TN + 4], com_a[0].y, com_b[1]);     // index_q = 1, 4<= index_v <= 7
        addSpecial(&tmp[2 * TN], com_a[0].z, com_b[0]);     // index_q = 2, 0<= index_v <= 3
        addSpecial(&tmp[2 * TN + 4], com_a[0].z, com_b[1]); // index_q = 2, 4<= index_v <= 7
        addSpecial(&tmp[3 * TN], com_a[0].w, com_b[0]);     // index_q = 3, 0<= index_v <= 3
        addSpecial(&tmp[3 * TN + 4], com_a[0].w, com_b[1]); // index_q = 3, 4<= index_v <= 7

        addSpecial(&tmp[4 * TN], com_a[1].x, com_b[0]);     // index_q = 4, 0<= index_v <= 3
        addSpecial(&tmp[4 * TN + 4], com_a[1].x, com_b[1]); // index_q = 4, 4<= index_v <= 7
        addSpecial(&tmp[5 * TN], com_a[1].y, com_b[0]);     // index_q =5, 0<= index_v <= 3
        addSpecial(&tmp[5 * TN + 4], com_a[1].y, com_b[1]); // index_q = 5, 4<= index_v <= 7
        addSpecial(&tmp[6 * TN], com_a[1].z, com_b[0]);     // index_q = 6, 0<= index_v <= 3
        addSpecial(&tmp[6 * TN + 4], com_a[1].z, com_b[1]); // index_q = 6, 4<= index_v <= 7
        addSpecial(&tmp[7 * TN], com_a[1].w, com_b[0]);     // index_q =7, 0<= index_v <= 3
        addSpecial(&tmp[7 * TN + 4], com_a[1].w, com_b[1]); // index_q = 7, 4<= index_v <= 7
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        (float4 &)dC[(indA + threadIdx.x * TM + index_q) * N + indB + threadIdx.y * TN] = (float4 &)tmp[index_q * TN];
        (float4 &)dC[(indA + threadIdx.x * TM + index_q) * N + indB + threadIdx.y * TN + 4] = (float4 &)tmp[index_q * TN + 4];
    }
}
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matrixOrigin(float *dA, float *dB, float *dC, int M, int K, int N)
{

    int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
    int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
    float tmp[TM][TN] = {0.0f};
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            if (indA + index_q < M && indB + index_v < N)
            {
                for (int s = 0; s < K; s++)
                {
                    tmp[index_q][index_v] += dA[(indA + index_q) * K + s] * dB[s * N + indB + index_v];
                }
            }
        }
    }
    for (int index_q = 0; index_q < TM; index_q++)
    {
        for (int index_v = 0; index_v < TN; index_v++)
        {
            if (indA + index_q < M && indB + index_v < N)
            {
                dC[(indA + index_q) * N + indB + index_v] = tmp[index_q][index_v];
            }
        }
    }
}

void hostMatrix(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    cudaSetDevice(3);
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    int repeat = 20;
    matrixKernel3th<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N); // warm up
    // matrixKernel4th<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        matrixKernel3th<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        // matrixKernel4th<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        //     matrixOrigin<BM, BN, BK, TM, TN><<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ela = get_walltime() - st;
    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main()
{
    float *hostA, *hostB, *hostC, *serialC;
    int M = 1024;
    int K = 1024;
    int N = 1024;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    serialC = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++)
    {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++)
    {
        hostB[i] = i % 3;
    }

    hostMatrix(hostA, hostB, hostC, M, K, N);
    double st, ela;
    st = get_walltime();
    matrixSerial(hostA, hostB, serialC, M, K, N);
    ela = get_walltime() - st;
    printf("CPU time:%.2f second\n", ela);
    compare(hostC, serialC, M, N);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}
