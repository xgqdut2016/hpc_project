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
#define BLOCK_DIM_x 32
#define BLOCK_DIM_y 32
#define max_function(a, b) ((a) > (b) ? (a) : (b))

__global__ void _attentionKernel(const float *inputQ, const float *inputK,
                                 const float *inputV, int N, int d,
                                 float *output) {
    int i = blockIdx.x;                              // i must < N,Q[i]
    int phd = threadIdx.y + blockIdx.y * blockDim.y; // V[:,d]
    int phNumN = (N + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    __shared__ float old_max[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float new_max[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float new_sum[BLOCK_DIM_x][BLOCK_DIM_y];
    old_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
    new_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
    new_sum[threadIdx.x][threadIdx.y] = 0.0f;
    __shared__ float block_sum[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float block_max[BLOCK_DIM_x][BLOCK_DIM_y];
    
    __shared__ float inputS[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float shareV[BLOCK_DIM_x][BLOCK_DIM_y];
    __shared__ float out[BLOCK_DIM_y];
    for (int phn = 0; phn < phNumN; phn++) {
        int j = threadIdx.x + phn * BLOCK_DIM_x;
        inputS[threadIdx.x][threadIdx.y] = 0.0f;
        block_max[threadIdx.x][threadIdx.y] = -__FLT_MAX__;
        block_sum[threadIdx.x][threadIdx.y] = 0.0f;
        shareV[threadIdx.x][threadIdx.y] = 0.0f;

        if (j < N && phd < d) {
            float sum_s = 0;
            for(int index = 0; index < d; index++){
                sum_s += inputQ[i * d + index] * inputK[j * d + index];
            }
            shareV[threadIdx.x][threadIdx.y] = inputV[j * d + phd];
            inputS[threadIdx.x][threadIdx.y] = sum_s;
            block_max[threadIdx.x][threadIdx.y] = sum_s;
            block_sum[threadIdx.x][threadIdx.y] = 1.0f;
        }

        __syncthreads();
        for (int strip = BLOCK_DIM_x / 2; strip > 0; strip = strip / 2) {
            if (threadIdx.x < strip) {
                if (block_max[threadIdx.x][threadIdx.y] >
                    block_max[threadIdx.x + strip][threadIdx.y]) {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x][threadIdx.y] +
                        block_sum[threadIdx.x + strip][threadIdx.y] *
                            __expf(block_max[threadIdx.x + strip][threadIdx.y] -
                                   block_max[threadIdx.x][threadIdx.y]);
                } else {
                    block_sum[threadIdx.x][threadIdx.y] =
                        block_sum[threadIdx.x + strip][threadIdx.y] +
                        block_sum[threadIdx.x][threadIdx.y] *
                            __expf(block_max[threadIdx.x][threadIdx.y] -
                                   block_max[threadIdx.x + strip][threadIdx.y]);
                    block_max[threadIdx.x][threadIdx.y] =
                        block_max[threadIdx.x + strip][threadIdx.y];
                }
            }
            __syncthreads();
        }
        __syncthreads();
        if (j < N) {
            if (new_max[threadIdx.x][threadIdx.y] > block_max[0][threadIdx.y]) {
                new_sum[threadIdx.x][threadIdx.y] =
                    new_sum[threadIdx.x][threadIdx.y] +
                    block_sum[0][threadIdx.y] *
                        __expf(block_max[0][threadIdx.y] -
                               new_max[threadIdx.x][threadIdx.y]);
            } else {
                new_sum[threadIdx.x][threadIdx.y] =
                    block_sum[0][threadIdx.y] +
                    new_sum[threadIdx.x][threadIdx.y] *
                        __expf(new_max[threadIdx.x][threadIdx.y] -
                               block_max[0][threadIdx.y]);
                new_max[threadIdx.x][threadIdx.y] = block_max[0][threadIdx.y];
            }
        }

        __syncthreads();

        if (j < N) {
            inputS[threadIdx.x][threadIdx.y] =
                __expf(inputS[threadIdx.x][threadIdx.y] -
                       new_max[threadIdx.x][threadIdx.y]);
            
        } else {
            inputS[threadIdx.x][threadIdx.y] = 0.0f;
            
        }
        __syncthreads();

        
        float sum_o = 0.0f;
        for (int index = 0; index < BLOCK_DIM_x; index++) {
                
            sum_o += inputS[index][threadIdx.y] *
                             shareV[index][threadIdx.y];
                
        }
        if (phn == 0) {
            out[threadIdx.y] = sum_o;
                //output[i * d + phd] = sum_o;
        } else {
            out[threadIdx.y] =
                    __expf(old_max[threadIdx.x][threadIdx.y] -
                           new_max[threadIdx.x][threadIdx.y]) *
                        out[threadIdx.y] +
                    sum_o;
        }

        old_max[threadIdx.x][threadIdx.y] =
                new_max[threadIdx.x][threadIdx.y];
        __syncthreads();
    }
    __syncthreads();
    if (phd < d)
        output[i * d + phd] =
            out[threadIdx.y] *
            __fdividef(1.0F, new_sum[threadIdx.x][threadIdx.y]);
}
void attention(float *cpu_Q, float *cpu_K, float *cpu_V, int N, int d, float *cpu_output){
    double st, ela;
    st = get_walltime();
    
    float *Q, *K, *V, *S, *output;
    cudaMalloc((void **) &Q, N*d*sizeof(float));
    cudaMalloc((void **) &K, N*d*sizeof(float));
    cudaMalloc((void **) &V, N*d*sizeof(float));
    cudaMalloc((void **) &S, N*N*sizeof(float));
    
    cudaMalloc((void **) &output, N*d*sizeof(float));
    cudaMemcpy(Q, cpu_Q, N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K, cpu_K, N*d*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(V, cpu_V, N*d*sizeof(float), cudaMemcpyHostToDevice);
    
    
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    int num_block_x = N;
    int num_block_y = (d + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);
    //int share_mem = (7 * BLOCK_DIM_x) * BLOCK_DIM_y * sizeof(float);
    _attentionKernel<<<grid_dim, block_dim>>>(Q, K, V, N, d, output);
        
        
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    cudaMemcpy(cpu_output, output, N*d*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(S);
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






