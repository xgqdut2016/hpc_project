#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define max_iter 20000
#define BLOCK_DIM 16
extern "C" void cuda_solve(double *f_1d, double *u_old, double *u_new, double eps, double r1, double r2, double r3, double r, int M, int N);
// c++ language use c ,must extern "C"
__global__
void cuda_update(double *cuda_u_old, double *cuda_u_new, int M, int N){
    // must void type
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    
    
    if (j < N && i < M){
        cuda_u_old[j*M + i] = cuda_u_new[j*M + i];
    }
    
}
__global__
void cuda_Jacobi(double *cuda_f_1d, double *cuda_u_old, double *cuda_u_new, double r1, double r2, double r3, double r, int M, int N){
    // must void type
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    
    double resid = 0;
    
    if (j > 0 && j < N - 1 && i > 0 && i < M - 1){
        resid = cuda_f_1d[j*M + i] - \
        (r1*(cuda_u_old[(j - 1)*M + i - 1] + cuda_u_old[(j - 1)*M + i + 1]) + \
        r3*(cuda_u_old[j*M + i - 1] + cuda_u_old[j*M + i + 1]) + \
        r1*(cuda_u_old[(j + 1)*M + i - 1] + cuda_u_old[(j + 1)*M + i + 1]) + \
        r2*(cuda_u_old[(j - 1)*M + i] + cuda_u_old[(j + 1)*M + i]) + r*cuda_u_old[j*M + i]);
        cuda_u_new[j*M + i] = cuda_u_old[j*M + i] + resid/r;
                
    }
    
}


void cuda_solve(double *f_1d, double *u_old, double *u_new, double eps, double r1, double r2, double r3, double r, int M, int N){
    int size = (M*N)*sizeof(double);
    double *cuda_f_1d, *cuda_u_old, *cuda_u_new;
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //-------------------
    cudaMalloc((void **) &cuda_f_1d, size);
    cudaMalloc((void **) &cuda_u_old, size);
    cudaMalloc((void **) &cuda_u_new, size);

    cudaMemcpy(cuda_f_1d, f_1d, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_u_old, u_old, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_u_new, u_new, size, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(M/(double)(BLOCK_DIM)),ceil(N/(double)(BLOCK_DIM)));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

    int k = 0;
    
    cudaEventRecord(start,0);
    while (k < max_iter){
        cuda_update<<<grid_dim, block_dim>>>(cuda_u_old, cuda_u_new, M, N);
        cudaDeviceSynchronize();// must wait
        cuda_Jacobi<<<grid_dim, block_dim>>>(cuda_f_1d,cuda_u_old, cuda_u_new, r1, r2, r3, r, M, N);
        
        k += 1;
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time

    cudaMemcpy(f_1d, cuda_f_1d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(u_old, cuda_u_old, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(u_new, cuda_u_new, size, cudaMemcpyDeviceToHost);

    cudaFree(cuda_f_1d);
    cudaFree(cuda_u_old);
    cudaFree(cuda_u_new);
    printf("grid dim: %d, %d\n",grid_dim.x, grid_dim.y);
    printf("block dim: %d, %d\n",block_dim.x, block_dim.y);
    printf("kernel launch time:%.5f\n",ker_time/1000.);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

