#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>



double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}
template <int BLOCK_DIM_x, int BLOCK_DIM_y>
__global__ void _attentionKernel(const float *__restrict inputQ,
                                 const float *__restrict inputK,
                                 const float *__restrict inputV, int N, int d,
                                 float *__restrict output) {
    int i = blockIdx.y;                              // i must < N,Q[i]
    int phd = threadIdx.x + blockIdx.x * blockDim.x; // V[:,d]

    int phNumN = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    __shared__ float inputS[BLOCK_DIM_x][BLOCK_DIM_y];
    float newMax;
    float oldMax;
    float newSum;

    newMax = -__FLT_MAX__;
    oldMax = -__FLT_MAX__;
    newSum = 0.0f;

    float out;
    out = 0.0f;
    //---------
    __shared__ float block_sum[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float block_max[BLOCK_DIM_x][BLOCK_DIM_y];

    for (int phn = 0; phn < phNumN; phn++) {

        int j = threadIdx.y + phn * BLOCK_DIM_y;
        float sum_s = 0.0f;
        for (int index = 0; index < d; index++) {
            sum_s += inputQ[i * d + index] * inputK[j * d + index];
        }

        if (j < N) {

            block_max[threadIdx.x][threadIdx.y] = sum_s;
            block_sum[threadIdx.x][threadIdx.y] = 1.0f;
        } else {

            block_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
            block_sum[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();
        for (int strip = BLOCK_DIM_y / 2; strip > 0; strip /= 2) {
            if (threadIdx.y < strip) {
                if (block_max[threadIdx.x][threadIdx.y] >
                    block_max[threadIdx.x][threadIdx.y + strip]) {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x][threadIdx.y] +
                        block_sum[threadIdx.x][threadIdx.y + strip] *
                            __expf(block_max[threadIdx.x][threadIdx.y + strip] -
                                   block_max[threadIdx.x][threadIdx.y]);
                } else {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x][threadIdx.y + strip] +
                        block_sum[threadIdx.x][threadIdx.y] *
                            __expf(block_max[threadIdx.x][threadIdx.y] -
                                   block_max[threadIdx.x][threadIdx.y + strip]);
                    block_max[threadIdx.x][threadIdx.y] =
                        block_max[threadIdx.x][threadIdx.y + strip];
                }
            }
            __syncthreads();
        }
        if (newMax > block_max[threadIdx.x][0]) {
            newSum = newSum + block_sum[threadIdx.x][0] *
                                  __expf(block_max[threadIdx.x][0] - newMax);
        } else {
            newSum = block_sum[threadIdx.x][0] +
                     newSum * __expf(newMax - block_max[threadIdx.x][0]);
            newMax = block_max[threadIdx.x][0];
        }

        __syncthreads();
        if (j < N && phd < d) {
            inputS[threadIdx.x][threadIdx.y] = __expf(sum_s - newMax);
        } else {
            inputS[threadIdx.x][threadIdx.y] = 0.0f;
        }
        __syncthreads();
        float sum_o = 0.0f;
        for (int index = 0; index < BLOCK_DIM_y; index++) {
            sum_o += inputS[threadIdx.x][index] *
                     inputV[(index + phn * BLOCK_DIM_y) * d + phd];
        }
        if (j < N && phd < d) {
            out = __expf(oldMax - newMax) * out + sum_o;
        }
        oldMax = newMax;
        __syncthreads();
    }

    if (threadIdx.y + (phNumN - 1) * BLOCK_DIM_y < N && phd < d) {
        output[i * d + phd] = out * __fdividef(1.0F, newSum);
    }
}
void attention(float *cpu_Q, float *cpu_K, float *cpu_V, int N, int d, float *cpu_output){
    double st, ela;
    st = get_walltime();
    
    float *inputQ, *inputK, *inputV, *output;
    cudaMalloc((void **) &inputQ, N*d*sizeof(float));
    cudaMalloc((void **) &inputK, N*d*sizeof(float));
    cudaMalloc((void **) &inputV, N*d*sizeof(float));
    
    
    cudaMalloc((void **) &output, N*d*sizeof(float));
    cudaMemcpy(inputQ, cpu_Q, N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inputK, cpu_K, N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inputV, cpu_V, N*d*sizeof(float), cudaMemcpyHostToDevice);
    
    
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    int num_block_y = N;
    if (d > 512) {
        int BLOCK_DIM_x = 1024;
        int BLOCK_DIM_y = 1;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<1024, 1>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 256) {
        int BLOCK_DIM_x = 512;
        int BLOCK_DIM_y = 2;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<512, 2>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 128) {
        int BLOCK_DIM_x = 256;
        int BLOCK_DIM_y = 4;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<256, 4>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 64) {
        int BLOCK_DIM_x = 128;
        int BLOCK_DIM_y = 8;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<128, 8>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 32) {
        int BLOCK_DIM_x = 64;
        int BLOCK_DIM_y = 16;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<64, 16>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 16) {
        int BLOCK_DIM_x = 32;
        int BLOCK_DIM_y = 32;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<32, 32>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else {
        int BLOCK_DIM_x = 16;
        int BLOCK_DIM_y = 64;
        int num_block_x = (d + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
        dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<16, 64>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    }
        
        
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    cudaMemcpy(cpu_output, output, N*d*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(inputQ);
    cudaFree(inputK);
    cudaFree(inputV);
    
    cudaFree(output);
    
    ela = get_walltime() - st;
    
    printf("BlockReduce,kernel time:%.4f, use time:%.4f\n", ker_time/1000., ela);
    
}
int main() {
    int N = 4;
    int d = 3;
    
    int size = N*d;
    
    
    float *cpu_Q, *cpu_K, *cpu_V, *cpu_output;
    cpu_Q = (float *)malloc(size*sizeof(float));
    cpu_K = (float *)malloc(size*sizeof(float));
    cpu_V = (float *)malloc(size*sizeof(float));
    cpu_output = (float *)malloc(size*sizeof(float));
    for(int i = 0; i < size; i++){
        cpu_Q[i] = i%4;
        cpu_K[i] = i%4;
        cpu_V[i] = i%4;
        //printf("Q:%.4f\n",cpu_Q[i]);
    }
    
    
    attention(cpu_Q, cpu_K, cpu_V, N, d, cpu_output);
    for(int i = 0; i < 10; i++){
        
        printf("out:%.6e\n",cpu_output[i]);
    }
    
    
    free(cpu_Q);
    free(cpu_K);
    free(cpu_V);
    free(cpu_output);
    
    return 0;
}







