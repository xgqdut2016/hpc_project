#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#define BLOCK_DIM 256
#define max_iter 20000

extern "C" void cuda_solve(const int *row_ptr, const int *col_ind, const double *values, const int num_rows, int num_vals, double *u_new, double *rig);
    
__global__
void cuda_update(double *cuda_u_old, double *cuda_u_new, int num_rows){
    // must void type
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (i < num_rows){
        cuda_u_old[i] = cuda_u_new[i];
    }
    
}
__global__
void cuda_Jacobi(const int *cuda_row_ptr, const int *cuda_col_ind, const double *cuda_values, const int num_rows, double *cuda_u_new,double *cuda_u_old, const double *cuda_rig){
    
    double tmp;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_rows){
        double sum = 0;
        const int row_start = cuda_row_ptr[i];
        const int row_end = cuda_row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += cuda_values[j] * cuda_u_old[cuda_col_ind[j]];
            if (cuda_col_ind[j] == i){
                tmp = cuda_values[j];
            }
        }
        cuda_u_new[i] = cuda_u_old[i] + (cuda_rig[i] - sum)/tmp;
    }
       
}

void cuda_solve(const int *row_ptr, const int *col_ind, const double *values, const int num_rows, int num_vals, double *u_new, double *rig){
    int size0 = num_rows*sizeof(double);
    int size1 = (num_rows + 1)*sizeof(int);
    int size2 = (num_vals)*sizeof(double);
    int size3 = (num_vals)*sizeof(int);
    //------------------------------
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double *cuda_values, *cuda_u_new,*cuda_u_old, *cuda_rig;
    cudaMalloc((void **) &cuda_u_new, size0);
    cudaMalloc((void **) &cuda_u_old, size0);
    cudaMalloc((void **) &cuda_rig, size0);
    cudaMalloc((void **) &cuda_values, size2);

    int *cuda_row_ptr, *cuda_col_ind;
    cudaMalloc((void **) &cuda_row_ptr, size1);// int 
    cudaMalloc((void **) &cuda_col_ind, size3);//int 

    cudaMemcpy(cuda_u_new, u_new, size0, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_rig, rig, size0, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_row_ptr, row_ptr, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_col_ind, col_ind, size3, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_values, values, size2, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(num_rows/(double)(BLOCK_DIM)),1,1);
    dim3 block_dim(BLOCK_DIM,1,1);
    
    cudaEventRecord(start,0);// start time
    int k;
    k = 0;
    while (k < max_iter){
        cuda_update<<<grid_dim, block_dim>>>(cuda_u_old, cuda_u_new, num_rows);
        cudaDeviceSynchronize();// must wait
        cuda_Jacobi<<<grid_dim, block_dim>>>(cuda_row_ptr, cuda_col_ind, cuda_values, num_rows, cuda_u_new, cuda_u_old, cuda_rig);
        k += 1;
    }
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time

    cudaMemcpy(u_new, cuda_u_new, size0, cudaMemcpyDeviceToHost);

    cudaFree(cuda_u_new);
    cudaFree(cuda_u_old);
    cudaFree(cuda_rig);
    cudaFree(cuda_row_ptr);
    cudaFree(cuda_col_ind);
    cudaFree(cuda_values);
    printf("grid dim: %d\n",grid_dim.x);
    printf("block dim: %d\n",block_dim.x);
    printf("kernel launch time:%.5f\n",ker_time/1000.);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}


