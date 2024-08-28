#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define BLOCK_DIM ((int)8)
extern "C" void cuda_jacobi_solve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N);
extern "C" double cuda_norm(double *cuda_u, int M, int N);
extern "C" void device_jacobi(int epoch, double eps,double r1, double r2, double r, double *cuda_u_old, double *cuda_u_new, double *cuda_b, double *cuda_resid, int M, int N);
extern "C" void cuda_twosolve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N);
// c++ language use c ,must extern "C"
__global__
void cuda_copy(double *cuda_u_new, double *cuda_u_old, int M, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    if(j < N + 1 && i < M + 1){
        cuda_u_old[j*(M + 1) + i] = cuda_u_new[j*(M + 1) + i];
    }
}
__global__
void cuda_jacobi(double r1,double r2,double r, double *cuda_u_old, double *cuda_u_new, double *cuda_b, double *cuda_resid, int M, int N){
    
    double temp;
    
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    if (j < N + 1 && i < M + 1){
        if(j == 0 || j == N || i == 0 || i == M){
            cuda_u_new[j*(M + 1) + i] = cuda_b[j*(M + 1) + i];
        }
        else {
            temp = cuda_b[j*(M + 1) + i] + \
            r1*(cuda_u_old[j*(M + 1) + i - 1] + cuda_u_old[j*(M + 1) + i + 1]) + \
            r2*(cuda_u_old[(j - 1)*(M + 1) + i] + cuda_u_old[(j + 1)*(M + 1) + i]) - \
            r*cuda_u_old[j*(M + 1) + i];
            cuda_u_new[j*(M + 1) + i] = cuda_u_old[j*(M + 1) + i] + temp/r;
            cuda_resid[j*(M + 1) + i] = temp;
        }
    } 
}
void cuda_jacobi_solve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N){
    double dx = (bound[0][1] - bound[0][0])/M;
    double dy = (bound[1][1] - bound[1][0])/N;
    double r1 = dy/dx, r2 = dx/dy;
    double r = 2*(r1 + r2);

    int size = (M + 1)*(N + 1)*sizeof(double);
    double *cuda_b, *cuda_u_old, *cuda_u_new, *cuda_resid;
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //-------------------
    cudaMalloc((void **) &cuda_b, size);
    cudaMalloc((void **) &cuda_u_old, size);
    cudaMalloc((void **) &cuda_u_new, size);
    cudaMalloc((void **) &cuda_resid, size);

    cudaMemcpy(cuda_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_u_old, u_old, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_u_new, u_new, size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_resid, resid, size, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil((M + 1)/(double)(BLOCK_DIM)),ceil((N + 1)/(double)(BLOCK_DIM)));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);

    int k = 0;
    
    cudaEventRecord(start,0);
    while (k < epoch){
        
        cuda_jacobi<<<grid_dim, block_dim>>>(r1,r2,r, cuda_u_old, cuda_u_new, cuda_b, cuda_resid, M, N);
        cudaDeviceSynchronize();// must wait
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess && k == 0) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
    // 处理CUDA错误
        }
        
        cuda_copy<<<grid_dim, block_dim>>>(cuda_u_new, cuda_u_old, M, N);
        
        k += 1;
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    double res = cuda_norm(cuda_resid, M, N);
    cudaMemcpy(b, cuda_b, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(u_old, cuda_u_old, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(u_new, cuda_u_new, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(resid, cuda_resid, size, cudaMemcpyDeviceToHost);

    cudaFree(cuda_b);
    cudaFree(cuda_u_old);
    cudaFree(cuda_u_new);
    cudaFree(cuda_resid);
    printf("grid dim: %d, %d\n",grid_dim.x, grid_dim.y);
    printf("block dim: %d, %d\n",block_dim.x, block_dim.y);
    printf("kernel launch time:%.5f,res:%.3e\n",ker_time/1000.,res);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //----------------
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s \n",cudaGetErrorString(err));
    }

    size_t free, total;
    cudaMemGetInfo(&free,&total);
    printf("Free memory: %zu\n Total memory: %zu\n", free, total);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    printf("Max threads per block: %d \n Max grid size: %d,%d,%d \n",
    prop.maxThreadsPerBlock,prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
    /***
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr,myKernel);
    printf("Max dynamic shared memory size: %zu\n",attrsharedSizeBytes);
    ***/
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s \n", cudaGetErrorString(err));
    }
}
//----------------------------------------------------------------
__global__
void cuda_init0(double *cuda_u, int M, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    if(j < N + 1 && i < M + 1){
        cuda_u[j*(M + 1) + i] = 0;
    }
}
__global__
void cuda_low(int M, int N, double *cuda_u_long, double *cuda_u_short){
    int m = M/2;
    int n = N/2;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    //----------------
    cuda_u_short[0] = cuda_u_long[0];
    cuda_u_short[m] = cuda_u_long[M];
    cuda_u_short[n*(m + 1)] = cuda_u_long[N*(M + 1)];
    cuda_u_short[n*(m + 1) + m] = cuda_u_long[N*(M + 1) + M];
    double temp = 0;
    //--------------------
    if (j > 0 && j < n){
        cuda_u_short[j*(m + 1)] = 0.5*(cuda_u_long[(2*j - 1)*(M + 1)] + cuda_u_long[(2*j + 1)*(M + 1)]);
        cuda_u_short[j*(m + 1) + m] = 0.5*(cuda_u_long[(2*j - 1)*(M + 1) + M] + cuda_u_long[(2*j + 1)*(M + 1) + M]);
    }
    else if(i > 0 && i < m){
        cuda_u_short[i] = 0.5*(cuda_u_long[2*i - 1] + cuda_u_long[2*i + 1]);
        cuda_u_short[n*(m + 1) + i] = 0.5*(cuda_u_long[N*(M + 1) + 2*i - 1] + cuda_u_long[N*(M + 1) + 2*i + 1]);
    }
    else if(j > 0 && j < n && i > 0 && i < m){
        temp = cuda_u_long[(2*j - 1)*(M + 1) + 2*i - 1] + cuda_u_long[(2*j - 1)*(M + 1) + 2*i + 1] + \
            cuda_u_long[(2*j + 1)*(M + 1) + 2*i - 1] + cuda_u_long[(2*j + 1)*(M + 1) + 2*i + 1] + \
            2*(cuda_u_long[(2*j - 1)*(M + 1) + 2*i] + cuda_u_long[(2*j + 1)*(M + 1) + 2*i]) + \
            2*(cuda_u_long[2*j*(M + 1) + 2*i - 1] + cuda_u_long[2*j*(M + 1) + 2*i + 1]) + \
            4*cuda_u_long[2*j*(M + 1) + 2*i];
        cuda_u_short[j*(m + 1) + i] = temp/16.0;
    }
}
__global__
void cuda_high(int m, int n, double *cuda_u_short, double *cuda_u_long){
    int M = 2*m;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    if(j < n && i < m){
        cuda_u_long[2*j*(M + 1) + 2*i] = cuda_u_short[j*(m + 1) + i];
        cuda_u_long[(2*j + 1)*(M + 1) + 2*i] = 0.5*(cuda_u_short[j*(m + 1) + i] + cuda_u_short[(j + 1)*(m + 1) + i]);
        cuda_u_long[2*j*(M + 1) + 2*i + 1] = 0.5*(cuda_u_short[j*(m + 1) + i] + cuda_u_short[j*(m + 1) + i + 1]);
        cuda_u_long[(2*j + 1)*(M + 1) + 2*i + 1] = 0.25*\
            (cuda_u_short[(j + 1)*(m + 1) + i] + cuda_u_short[(j + 1)*(m + 1) + i + 1] + cuda_u_short[j*(m + 1) + i + 1] + cuda_u_short[j*(m + 1) + i]);
    }
}
__global__
void cuda_plus(double *cuda_y, double *cuda_x, double a, int M, int N){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    if(j < N + 1 && i < M + 1){
        cuda_y[j*(M + 1) + i] = cuda_y[j*(M + 1) + i] + a*cuda_x[j*(M + 1) + i];
    }
}
double cuda_norm(double *cuda_u, int M, int N){
    double f = 0;
    int size = (M + 1)*(N + 1)*sizeof(double);
    double *host_u;
    host_u = (double *)malloc(size);
    cudaMemcpy(host_u, cuda_u, size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < (M + 1)*(N + 1); i++){
        f += pow(host_u[i], 2);
    }
    return sqrt(f/((M + 1)*(N + 1)));
}
void device_jacobi(int epoch, double eps,double r1, double r2, double r, double *cuda_u_old, double *cuda_u_new, double *cuda_b, double *cuda_resid, int M, int N){
    int k = 0;
    dim3 grid_dim(ceil((M + 1)/(double)(BLOCK_DIM)),ceil((N + 1)/(double)(BLOCK_DIM)));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    while (k < epoch){
        cuda_jacobi<<<grid_dim, block_dim>>>(r1,r2,r, cuda_u_old, cuda_u_new, cuda_b, cuda_resid, M, N);
        cudaDeviceSynchronize();// must wait
        cuda_copy<<<grid_dim, block_dim>>>(cuda_u_new, cuda_u_old, M, N);
        k += 1;
    }
}
void cuda_twosolve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N){
    double dx = (bound[0][1] - bound[0][0])/M;
    double dy = (bound[1][1] - bound[1][0])/N;
    double r1 = dy/dx, r2 = dx/dy;
    double r = 2*(r1 + r2);
    int gepoch = 20;
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //-------------------
    int cuda_size = (M + 1)*(N + 1)*sizeof(double);
    double *cuda_u_old, *cuda_u_new, *cuda_b, *cuda_resid;

    cudaMalloc((void **) &cuda_b, cuda_size);
    cudaMalloc((void **) &cuda_resid, cuda_size);
    cudaMalloc((void **) &cuda_u_old, cuda_size);
    cudaMalloc((void **) &cuda_u_new, cuda_size);

    cudaMemcpy(cuda_b, b, cuda_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_resid, resid, cuda_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_u_old, u_old, cuda_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_u_new, u_new, cuda_size, cudaMemcpyHostToDevice);
    //-------------------
    int m = M/2,n = N/2;
    int sizesmall = (m + 1)*(n + 1)*sizeof(double);
    double *cuda_b_short, *cuda_u_short, *cuda_u_old_short, *cuda_resid_short;

    cudaMalloc((void **) &cuda_b_short, sizesmall);
    cudaMalloc((void **) &cuda_resid_short, sizesmall);
    cudaMalloc((void **) &cuda_u_short, sizesmall);
    cudaMalloc((void **) &cuda_u_old_short, sizesmall);

    dim3 grid_dim(ceil((M + 1)/(double)(BLOCK_DIM)),ceil((N + 1)/(double)(BLOCK_DIM)));
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    double norm = 0;
    cudaEventRecord(start,0);
    int k = 0;
    while(k < epoch){
        device_jacobi(gepoch, eps, r1,r2,r, cuda_u_old, cuda_u_new, cuda_b, cuda_resid, M, N);
        norm = cuda_norm(cuda_u_new, M, N);
        cuda_low<<<grid_dim, block_dim>>>(M, N, cuda_resid, cuda_b_short);
        cudaDeviceSynchronize();// must wait
        cuda_init0<<<grid_dim, block_dim>>>(cuda_u_old_short, m, n);
        cudaDeviceSynchronize();// must wait
        device_jacobi(gepoch, eps, r1,r2,r, cuda_u_old_short, cuda_u_short, cuda_b_short, cuda_resid_short, m,n);
        cuda_init0<<<grid_dim, block_dim>>>(cuda_u_old,M,N);
        cudaDeviceSynchronize();// must wait
        cuda_high<<<grid_dim, block_dim>>>(m, n, cuda_u_short, cuda_u_old);
        cudaDeviceSynchronize();// must wait
        cuda_plus<<<grid_dim, block_dim>>>(cuda_u_new, cuda_u_old, 1.0, M, N);
        cudaDeviceSynchronize();// must wait
        cuda_copy<<<grid_dim, block_dim>>>(cuda_u_new, cuda_u_old, M, N);
        cudaDeviceSynchronize();// must wait
        k += 1;
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    norm = cuda_norm(cuda_resid, M, N);
    cudaMemcpy(b, cuda_b, cuda_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(resid, cuda_resid, cuda_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(u_old, cuda_u_old, cuda_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(u_new, cuda_u_new, cuda_size, cudaMemcpyDeviceToHost);

    cudaFree(cuda_b);
    cudaFree(cuda_resid);
    cudaFree(cuda_u_old);
    cudaFree(cuda_u_new);
    cudaFree(cuda_resid_short);
    cudaFree(cuda_b_short);
    cudaFree(cuda_u_old_short);
    cudaFree(cuda_u_short);
    printf("grid dim: %d, %d\n",grid_dim.x, grid_dim.y);
    printf("block dim: %d, %d\n",block_dim.x, block_dim.y);
    printf("kernel launch time:%.5f,resid:%.3e\n",ker_time/1000.,norm);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

