#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

#define max_function(a, b) ((a) > (b) ? (a) : (b))

double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}

template <int BLOCK_DIM_y>
__launch_bounds__(BLOCK_DIM_y) __global__
    void _attentionKernel(const float *__restrict inputQ,
                          const float *__restrict inputK,
                          const float *__restrict inputV, int N, int d,
                          float *__restrict output) {
    int i = blockIdx.x;                              // i must < N,Q[i]
    int phd = threadIdx.y + blockIdx.y * blockDim.y; // V[:,d]

    __shared__ float old_max[BLOCK_DIM_y];
    __shared__ float new_max[BLOCK_DIM_y];
    __shared__ float new_sum[BLOCK_DIM_y];
    old_max[threadIdx.y] = -__FLT_MAX__;
    new_max[threadIdx.y] = -__FLT_MAX__;
    new_sum[threadIdx.y] = 0.0f;
    __shared__ float block_sum[BLOCK_DIM_y];
    __shared__ float block_max[BLOCK_DIM_y];

    __shared__ float inputS[BLOCK_DIM_y];
    __shared__ float shareV[BLOCK_DIM_y];
    __shared__ float out[BLOCK_DIM_y];

    int phNumD = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    __shared__ float shareQ_times_K[BLOCK_DIM_y];
    
    for (int phn = 0; phn < N; phn++) {
        shareV[threadIdx.y] = 0.0f;

        float sum_s = 0.0f;
        for (int ind = 0; ind < phNumD; ind++) {
            if (threadIdx.y + ind * BLOCK_DIM_y < d) {
                shareQ_times_K[threadIdx.y] =
                    inputQ[i * d + threadIdx.y + ind * BLOCK_DIM_y] * inputK[phn * d + threadIdx.y + ind * BLOCK_DIM_y];
                
            } else {
                shareQ_times_K[threadIdx.y] = 0.0f;
                
            }
            __syncthreads();
            for(int strip = BLOCK_DIM_y/2; strip > 0; strip = strip/2){
                if(threadIdx.y < strip){
                    shareQ_times_K[threadIdx.y] += shareQ_times_K[threadIdx.y + strip];
                }
                __syncthreads();
            }
            sum_s += shareQ_times_K[0];
            __syncthreads();
        }

        inputS[threadIdx.y] = sum_s;
        block_max[threadIdx.y] = sum_s;
        block_sum[threadIdx.y] = 1.0f;

        if (phd < d) {
            shareV[threadIdx.y] = inputV[phn * d + phd];
        }

        __syncthreads();

        if (new_max[threadIdx.y] > block_max[threadIdx.y]) {
            new_sum[threadIdx.y] =
                new_sum[threadIdx.y] +
                block_sum[threadIdx.y] *
                    __expf(block_max[threadIdx.y] - new_max[threadIdx.y]);
        } else {
            new_sum[threadIdx.y] =
                block_sum[threadIdx.y] +
                new_sum[threadIdx.y] *
                    __expf(new_max[threadIdx.y] - block_max[threadIdx.y]);
            new_max[threadIdx.y] = block_max[threadIdx.y];
        }

        __syncthreads();

        inputS[threadIdx.y] =
            __expf(inputS[threadIdx.y] - new_max[threadIdx.y]);

        __syncthreads();

        if (phn == 0) {
            out[threadIdx.y] = inputS[threadIdx.y] * shareV[threadIdx.y];

        } else {
            out[threadIdx.y] =
                __expf(old_max[threadIdx.y] - new_max[threadIdx.y]) *
                    out[threadIdx.y] +
                inputS[threadIdx.y] * shareV[threadIdx.y];
        }

        old_max[threadIdx.y] = new_max[threadIdx.y];

        __syncthreads();
    }
    __syncthreads();
    if (phd < d)
        output[i * d + phd] =
            out[threadIdx.y] * __fdividef(1.0F, new_sum[threadIdx.y]);
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
    int num_block_x = N;

    if (d > 1023) {
        int BLOCK_DIM_y = 1024;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<1024>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 511) {
        int BLOCK_DIM_y = 512;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<512>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 255) {
        int BLOCK_DIM_y = 256;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<256>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 127) {
        int BLOCK_DIM_y = 128;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<128>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 63) {
        int BLOCK_DIM_y = 64;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<64>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else if (d > 31) {
        int BLOCK_DIM_y = 32;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<32>
            <<<grid_dim, block_dim>>>(inputQ, inputK, inputV, N, d, output);
    } else {
        int BLOCK_DIM_y = 16;
        int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
        dim3 block_dim(1, BLOCK_DIM_y, 1);
        dim3 grid_dim(num_block_x, num_block_y, 1);
        _attentionKernel<16>
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
    int N = 1024;
    int d = 1000;
    
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






