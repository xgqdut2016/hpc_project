#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define max_function(a,b) ((a)>(b)?(a):(b))
#define N 32
#define max_iter 20000

double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}
float u_acc(float x,float y){
    return (1.0 - pow(x,2))*(1.0 - pow(y,2));
}
float f(float x,float y){
    
    return 2*(1.0 - pow(x,2)) + 2*(1.0 - pow(y,2));
}
int main(int argc, char *argv[]){
    
    float dx, dy;
    int m = N,n = N;
    printf("run Jacobi with gird size: m = %d n = %d max_iter = %d ...\n", m, n, max_iter);

    double time0, time1;
    
    float bound[2][2] = {{-1.0,1.0},{-1.0,1.0}};
    int i,j;
    float x,y;
    /**
    float* phi_1d = (float *) malloc(sizeof(float) *  m * n);
    float (*phi)[n] = (float (*)[n]) phi_1d;
    float (*u)[n] = (float (*)[n]) phi_1d;
    float (*f_1d)[n] = (float (*)[n]) phi_1d;
    **/
    
    int nthreads,tid;
    float phi[m][n],f_1d[m][n],u[m][n];
    dx = ((bound[0][1] - bound[0][0]) / (m - 1));
    dy = ((bound[1][1] - bound[1][0])/ (n - 1)); 
    time0 = get_walltime();

    
    for (i = 0; i < m; i++) {
	    for (j = 0; j < n; j++) {
	        x = bound[0][0] + i*dx,y = bound[1][0] + j*dy;
            phi[i][j] = 0.0;
            u[i][j] = u_acc(x,y);
	        f_1d[i][j] = 0.0;
	    }
    }
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            x = bound[0][0] + i*dx,y = bound[1][0] + j*dy;
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1){
                phi[i][j] = u_acc(x,y);
            }
            else {
                f_1d[i][j] = f(x,y);
            }
        }
    }

    
//solve
    
    int k = 0;
    
    float error = 0,L1_err = 0,eps = 1e-11; 
    float r1 = -0.5,r2 = -pow(dx/dy,2),r3 = -pow(dy/dx,2),r = 2*(1 - r2 - r3);
    float resid;
    float u_1d[m][n];
    

    while (k < max_iter) {
	    error = 0;
        L1_err = 0;
	    #pragma omp parallel for default(shared) private(i,j) schedule(runtime)
	    
	        //该并行区只有一个for循环，使用#pragma omp parallel for可以自动分配线程处理任务
       	    for(i = 0; i < m; i++){
	            for (j = 0;j < n;j++){
		            u_1d[i][j] = phi[i][j];
		        }
	        }
	    
        #pragma omp parallel for default(shared) private(i,j,tid,resid) schedule(runtime) \
        reduction(+:error) 
            
            for(i = 0; i < m; i++){
		        for (j = 0; j < n; j++){
                    if (i == 0 || i == N - 1 || j == 0 || j == N - 1){
                        continue;
                    }
                    else {
                        resid = (u_1d[i - 1][j + 1]*r1 + u_1d[i][j + 1]*r2 + u_1d[i + 1][j + 1]*r1
                        + u_1d[i - 1][j]*r3 + u_1d[i][j]*r + u_1d[i + 1][j]*r3 
                        + u_1d[i - 1][j - 1]*r1 + u_1d[i][j - 1]*r2 + u_1d[i + 1][j - 1]*r1 - f_1d[i][j]*(pow(dx,2) + pow(dy,2)));
                        phi[i][j] = u_1d[i][j] - resid/r;
		                error += resid*resid;
                    }     
		        }
	        }
        
        #pragma omp parallel for reduction(max:L1_err) schedule(runtime) ordered
            
            
            for(i = 0; i < m; i++){
                for(j = 0; j < n; j++){
                    L1_err = max_function(L1_err, fabs(phi[i][j] - u[i][j]));
                }
                if (k == 1){
                    nthreads = omp_get_num_threads();//获取当前使用线程数目
                    tid = omp_get_thread_num();
                    #pragma omp ordered
                    {
                        printf("tid = %d,i = %d\n",tid,i);
                    }
                    
                }
            }
            
            
	    
	    error  = sqrt(error)/(m*n);
        if (error < eps)
        {
            break;
        }
        if (k%1000 == 0){
	        printf("%d iteration,error: %3e,L1_err:%.3e\n",k,error,L1_err);
            
	    }
	    k = k + 1;
        
    }
    time1 = get_walltime() - time0;
    
    printf("Finish at %d,use time:%.2f,use threads:%d, the resid:%.2e,the err:%.2e\n",k,time1,nthreads,error,L1_err);
    
    return 0;
}





