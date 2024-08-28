#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#define BLOCK_DIM 64

extern "C" double cuda_dot(double *cuda_x, double *cuda_y, double *cuda_z, double *host_z, int num_rows);
extern "C" void cuda_minres(const int *row_ptr, const int *col_ind, const double *values, const int num_rows, int num_vals, double *u_new, double *rig);


__global__
void cuda_spmv_csr(const int *cuda_row_ptr, const int *cuda_col_ind, const double *cuda_values, const int num_rows, const double *cuda_x, double *cuda_y) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    
    if (i < num_rows){
        double sum = 0;
        const int row_start = cuda_row_ptr[i];
        const int row_end = cuda_row_ptr[i + 1];
        
        for (int j = row_start; j < row_end; j++) {
            sum += cuda_values[j] * cuda_x[cuda_col_ind[j]];
        }
        
        cuda_y[i] = sum;
    }

}
__global__
void cuda_init0(double *cuda_u, int num_rows){ // init all = 0
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < num_rows){
        cuda_u[i] = 0;
    }
}
__global__
void cuda_plus(double *cuda_x, double *cuda_y, double *cuda_z, int num_rows, double a){ // x + ay = z
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < num_rows){
        cuda_z[i] = cuda_x[i] + a*cuda_y[i];
    }
}
__global__
void cuda_times(double *cuda_x, double *cuda_z, int num_rows, double a){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i < num_rows){
        cuda_z[i] = a*cuda_x[i];
    }
}
__global__
void pre_dot(double *cuda_x, double *cuda_y, double *cuda_z, int num_rows){
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    
    if(i < num_rows){
        cuda_z[i] = cuda_x[i]*cuda_y[i];
    }

}
double cuda_dot(double *cuda_x, double *cuda_y, double *cuda_z, double *host_z, int num_rows){ //compute (x,y) dot
    double f = 0;
    int size = num_rows*sizeof(double);
    dim3 grid_dim(ceil(num_rows/(double)(BLOCK_DIM)),1,1);
    dim3 block_dim(BLOCK_DIM,1,1);
    pre_dot<<<grid_dim, block_dim>>>(cuda_x, cuda_y, cuda_z, num_rows);
    cudaDeviceSynchronize();// must wait
    cudaMemcpy(host_z, cuda_z, size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < num_rows; i++){
        f += host_z[i];
    }
    return f;
}
void cuda_minres(const int *row_ptr, const int *col_ind, const double *values, const int num_rows, int num_vals, double *u_new, double *rig){
    int size0 = num_rows*sizeof(double);
    int size1 = (num_rows + 1)*sizeof(int);
    int size2 = (num_vals)*sizeof(double);
    int size3 = (num_vals)*sizeof(int);
    //------------------------------
    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double *cuda_values, *cuda_u_new, *cuda_rig;
    cudaMalloc((void **) &cuda_u_new, size0);
    cudaMalloc((void **) &cuda_rig, size0);
    cudaMalloc((void **) &cuda_values, size2);

    int *cuda_row_ptr, *cuda_col_ind;
    cudaMalloc((void **) &cuda_row_ptr, size1);// int 
    cudaMalloc((void **) &cuda_col_ind, size3);//int 
    
    
    //-------------------
    

    cudaMemcpy(cuda_u_new, u_new, size0, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_rig, rig, size0, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_row_ptr, row_ptr, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_col_ind, col_ind, size3, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_values, values, size2, cudaMemcpyHostToDevice);

    dim3 grid_dim(ceil(num_rows/(double)(BLOCK_DIM)),1,1);
    dim3 block_dim(BLOCK_DIM,1,1);
    
    cudaEventRecord(start,0);// start time

    double eps = 1e-10;
    double *r,*w,*q_new,*q_old,*d_old,*d_mid,*d_new;
    cudaMalloc((void **) &r, size0);
    cudaMalloc((void **) &w, size0);
    cudaMalloc((void **) &q_old, size0);
    cudaMalloc((void **) &q_new, size0);
    cudaMalloc((void **) &d_old, size0);
    cudaMalloc((void **) &d_mid, size0);
    cudaMalloc((void **) &d_new, size0);
    double *cuda_z,host_z[num_rows];
    cudaMalloc((void **) &cuda_z, size0);
    for(int i = 0; i < num_rows; i++){
        host_z[i] = 0;
    }
    cuda_init0<<<grid_dim, block_dim>>>(r, num_rows);
    cuda_init0<<<grid_dim, block_dim>>>(w, num_rows);
    cuda_init0<<<grid_dim, block_dim>>>(q_old, num_rows);
    cuda_init0<<<grid_dim, block_dim>>>(q_new, num_rows);
    cuda_init0<<<grid_dim, block_dim>>>(d_old, num_rows);
    cuda_init0<<<grid_dim, block_dim>>>(d_mid, num_rows);
    cuda_init0<<<grid_dim, block_dim>>>(d_new, num_rows);
    cuda_init0<<<grid_dim, block_dim>>>(cuda_z, num_rows);
    cuda_spmv_csr<<<grid_dim, block_dim>>>(cuda_row_ptr, cuda_col_ind, cuda_values, num_rows, cuda_u_new, r);//r^0 = A@x^0
    cudaDeviceSynchronize();
    cuda_plus<<<grid_dim, block_dim>>>(cuda_rig, r, r, num_rows, -1);// r^0 = b - r^0
    cudaDeviceSynchronize();
    
    
    double a = sqrt(cuda_dot(r, r, cuda_z, host_z, num_rows));// a = sqrt(r^0,r^0)
    
    
    
    cudaDeviceSynchronize();
    cuda_times<<<grid_dim, block_dim>>>(r, q_new, num_rows, 1/a);// q_1 = r^0/a
    cudaDeviceSynchronize();
    
    double beta_old = 0,beta_new = 0;
    double r_old = 0,r_mid = 0,r_new = 0;
    double c_old = 0,c_mid = 0,c_new = 0;
    double s_old = 0,s_mid = 0,s_new = 0;
    double xi_old = a,xi_new = 0;

    
    double alpha = 0,alpha_hat = 0,beta_hat = 0,gamma = 0;
    
    for(int k = 0; k < num_rows; k++){
        cuda_spmv_csr<<<grid_dim, block_dim>>>(cuda_row_ptr, cuda_col_ind, cuda_values, num_rows, q_new, w);// w = A@q_new
        
        cudaDeviceSynchronize();
        cuda_times<<<grid_dim, block_dim>>>(q_old, q_old, num_rows, beta_old); // q_old = beta_old*q_old
        cudaDeviceSynchronize();
        cuda_plus<<<grid_dim, block_dim>>>(w, q_old, w, num_rows, -1);// w = w - q_old
        cudaDeviceSynchronize();
        alpha = cuda_dot(w, q_new, cuda_z, host_z, num_rows);
        cudaDeviceSynchronize();

        cuda_plus<<<grid_dim, block_dim>>>(w,q_new,w,num_rows,-alpha);// w = w - alpha*q_new
        cudaDeviceSynchronize();
        beta_new = sqrt(cuda_dot(w, w, cuda_z, host_z, num_rows));
        cudaDeviceSynchronize();
        
        if (k == 0){
            alpha_hat = alpha;
        }
        else if (k == 1){
            r_mid = c_mid*beta_old + s_mid*alpha;
            alpha_hat = -s_mid*beta_old + c_mid*alpha;
        }
        else {
            r_old = s_old*beta_old;
            beta_hat = c_old*beta_old;
            
            r_mid = c_mid*beta_hat + s_mid*alpha;
            alpha_hat = -s_mid*beta_hat + c_mid*alpha;
        }
        if (fabs(alpha_hat) > fabs(beta_new)){
            gamma = beta_new/alpha_hat;
            c_new = 1.0/sqrt(1 + gamma*gamma),s_new = c_new*gamma;
        }
            
        else{
            gamma = alpha_hat/beta_new;
            s_new = 1.0/sqrt(1 + gamma*gamma),c_new = s_new*gamma;
        }
        r_new = c_new*alpha_hat + s_new*beta_new;
        xi_new = -s_new*xi_old;
        xi_old = c_new*xi_old ;
        
        cuda_plus<<<grid_dim, block_dim>>>(q_new,d_old,d_new,num_rows, -r_old);// d_new = q_new - r_old*d_old
        cudaDeviceSynchronize();
        
        cuda_plus<<<grid_dim, block_dim>>>(d_new,d_mid,d_new,num_rows,-r_mid);// d_new -= r_mid*d_mid
        cudaDeviceSynchronize();
        
        cuda_times<<<grid_dim, block_dim>>>(d_new,d_new,num_rows,1.0/r_new);//d_new = d_new/r_new;
        cudaDeviceSynchronize();
        
        cuda_plus<<<grid_dim, block_dim>>>(cuda_u_new,d_new,cuda_u_new,num_rows,xi_old);
        cudaDeviceSynchronize();
        
        if(fabs(xi_new) < eps){
            break;
        }
        else {
            
            xi_old = xi_new;
            
            cuda_times<<<grid_dim, block_dim>>>(q_new,q_old,num_rows,1.0);
            cudaDeviceSynchronize();
            cuda_times<<<grid_dim, block_dim>>>(w,q_new,num_rows,1.0/beta_new);
            cudaDeviceSynchronize();
            
            beta_old = beta_new;
            c_old = c_mid;
            c_mid = c_new;
            s_old = s_mid;
            s_mid = s_new;
            
            r_old = r_mid;
            r_mid = r_new;
            
            cuda_times<<<grid_dim, block_dim>>>(d_mid,d_old,num_rows,1.0);
            cudaDeviceSynchronize();
            cuda_times<<<grid_dim, block_dim>>>(d_new,d_mid,num_rows,1.0);
            cudaDeviceSynchronize();
        }
    }

    if(fabs(xi_new) < eps){
        printf("cuda success\n");
    }
    else {
        printf("cuda fail\n");
    }
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time

    cudaMemcpy(u_new, cuda_u_new, size0, cudaMemcpyDeviceToHost);

    cudaFree(cuda_u_new);
    cudaFree(cuda_rig);
    cudaFree(cuda_row_ptr);
    cudaFree(cuda_col_ind);
    cudaFree(cuda_values);
    cudaFree(cuda_z);
    cudaFree(r);
    cudaFree(w);
    cudaFree(q_new);
    cudaFree(q_old);
    cudaFree(d_new);
    cudaFree(d_mid);
    cudaFree(d_old);

    printf("grid dim: %d\n",grid_dim.x);
    printf("block dim: %d\n",block_dim.x);
    printf("kernel launch time:%.5f\n",ker_time/1000.);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

