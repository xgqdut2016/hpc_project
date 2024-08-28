#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <mma.h>
using namespace nvcuda;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;
const int BLOCK_DIM_x = 16;
const int BLOCK_DIM_y = 16;
const int warpSize = 32;
const int warpNum = BLOCK_DIM_x * BLOCK_DIM_y / warpSize;
const int warpX = (warpNum == 1 ? 1 : 2);
const int warpY = warpNum / warpX;

__global__ void row_wmma_ker(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int lda = K; // A=[M,K],索引(x,y) = x * K + y，列优先原则索引(x,y) = y * M + x
    int ldb = K;
    int ldc = N;

    int indA = blockIdx.x * warpX * WMMA_M;
    int indB = blockIdx.y * warpY * WMMA_N;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    int aRow = indA + warpIdx * WMMA_M;
    int bCol = indB + warpIdy * WMMA_N;
    int width = (K + WMMA_K - 1) / WMMA_K;
    for (int i = 0; i < width; i++)
    {
        int aCol = i * WMMA_K;
        int bRow = i * WMMA_K;
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // 读取A,B矩阵里面子矩阵的元素
            wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(right_frag, dB + bRow + bCol * ldb, ldb);
            // 子矩阵做乘法
            wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
        }
    }
    int cRow = aRow;
    int cCol = bCol;
    if (cRow < M && cCol < N)
    {
        // Store the output
        wmma::store_matrix_sync(dC + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}
//-------
__device__ void wmmaBlock(float *dA, float *dB, float *dC, int indA, int indB, int M, int K, int N)
{
    int lda = K; // A=[M,K],索引(x,y) = x * K + y，列优先原则索引(x,y) = y * M + x
    int ldb = K;
    int ldc = N;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    int bCol = indB + warpIdx * WMMA_M;
    int aRow = indA + warpIdy * WMMA_N;
    int width = (K + WMMA_K - 1) / WMMA_K;
    for (int i = 0; i < width; i++)
    {
        int aCol = i * WMMA_K;
        int bRow = i * WMMA_K;
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // 读取A,B矩阵里面子矩阵的元素
            wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(right_frag, dB + bRow + bCol * ldb, ldb);
            // 子矩阵做乘法
            wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
        }
    }
    int cRow = aRow;
    int cCol = bCol;
    if (cRow < M && cCol < N)
    {
        // Store the output
        wmma::store_matrix_sync(dC + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
    }
}
__global__ void wmmaRowMatmul(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int indB = blockIdx.x * warpX * WMMA_M;
    int indA = blockIdx.y * warpY * WMMA_N;
    wmmaBlock(dA, dB, dC, indA, indB, M, K, N);
}
//-------
__device__ void wmmashareBlock(float *dA, float *dB, float *shareC, int indA, int indB, int M, int K, int N)
{
    int lda = K;              // 一个线程块内是[warpY * WMMA_N, K]
    int ldb = K;              // 一个线程块内是[K, warpX * WMMA_M]
    int ldc = warpX * WMMA_M; // 一个线程块内是[warpY * WMMA_N, warpX * WMMA_M]

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    int bCol = indB + warpIdx * WMMA_M;
    int aRow = indA + warpIdy * WMMA_N;
    int width = (K + WMMA_K - 1) / WMMA_K;
    for (int i = 0; i < width; i++)
    {
        int aCol = i * WMMA_K;
        int bRow = i * WMMA_K;
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // 读取A,B矩阵里面子矩阵的元素
            wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
            wmma::load_matrix_sync(right_frag, dB + bRow + bCol * ldb, ldb);
            // 子矩阵做乘法
            wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
        }
    }
    int cRow = warpIdy * WMMA_N;
    int cCol = warpIdx * WMMA_M;
    wmma::store_matrix_sync(shareC + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
}
__global__ void wmmashareRowMatmul(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int indB = blockIdx.x * warpX * WMMA_M;
    int indA = blockIdx.y * warpY * WMMA_N;
    __shared__ float shareC[warpY * WMMA_N * warpX * WMMA_M]; //[warpY * WMMA_N , warpX * WMMA_M]
    wmmashareBlock(dA, dB, shareC, indA, indB, M, K, N);
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    int cRowLocal = warpIdy * WMMA_N;
    int cColLocal = warpIdx * WMMA_M;
    int ldcLocal = warpX * WMMA_M;
    wmma::load_matrix_sync(c_frag, shareC + cRowLocal * ldcLocal + cColLocal, ldcLocal, wmma::mem_row_major);
    int cColGlobal = indB + warpIdx * WMMA_M;
    int cRowGlobal = indA + warpIdy * WMMA_N;
    int ldcGlobal = N;
    wmma::store_matrix_sync(dC + cRowGlobal * ldcGlobal + cColGlobal, c_frag, ldcGlobal, wmma::mem_row_major);
}

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void matrixSerial(float *hostA, float *hostB, float *hostC, int N, int d)
{
    float tmp = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            tmp = 0;
            for (int s = 0; s < d; s++)
            {
                tmp += hostA[i * d + s] * hostB[j * d + s];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
float compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
    }
    return error;
}

void hostMatrix(float *hostA, float *hostB, float *hostC, int N, int d)
{
    cudaSetDevice(9);
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, N * d * sizeof(float));
    cudaMalloc((void **)&dB, N * d * sizeof(float));
    cudaMalloc((void **)&dC, N * N * sizeof(float));

    cudaMemcpy(dA, hostA, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * d * sizeof(float), cudaMemcpyHostToDevice);

    int num_block_x = (N + WMMA_M * warpX - 1) / (WMMA_M * warpX);
    int num_block_y = (N + WMMA_N * warpY - 1) / (WMMA_N * warpY);

    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    cudaEvent_t start, stop;
    float ker_time = 0;

    // row_wmma_ker<<<grid_dim, block_dim>>>(dA, dB, dC, N, d, N);
    // wmmaRowMatmul<<<grid_dim, block_dim>>>(dA, dB, dC, N, d, N);
    wmmashareRowMatmul<<<grid_dim, block_dim>>>(dA, dB, dC, N, d, N);
    int repeat = 20;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        // row_wmma_ker<<<grid_dim, block_dim>>>(dA, dB, dC, N, d, N);
        // wmmaRowMatmul<<<grid_dim, block_dim>>>(dA, dB, dC, N, d, N);
        wmmashareRowMatmul<<<grid_dim, block_dim>>>(dA, dB, dC, N, d, N);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ela = get_walltime() - st;

    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}

int main()
{
    float *hostA, *hostB, *hostC, *serialC;
    int d = 512;

    int N = 2048;

    hostA = (float *)malloc(N * d * sizeof(float));
    hostB = (float *)malloc(N * d * sizeof(float));
    hostC = (float *)malloc(N * N * sizeof(float));
    serialC = (float *)malloc(N * N * sizeof(float));
    for (int i = 0; i < N * d; i++)
    {
        hostA[i] = i % 3;
        hostB[i] = i % 3;
    }

    double st, ela;
    st = get_walltime();
    matrixSerial(hostA, hostB, serialC, N, d);
    ela = get_walltime() - st;
    hostMatrix(hostA, hostB, hostC, N, d);
    float error = compare(hostC, serialC, N, N);
    printf("CPU time:%.2f, error:%.4e\n", ela, error);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}