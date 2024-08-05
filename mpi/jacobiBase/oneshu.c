#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include<time.h>
clock_t time0,time1;//clock_t为clock()函数返回的变量类型
#define max(a,b) ((a)>(b)?(a):(b))
#define N 128
#define max_iter 20000

float u_acc(float x,float y){
    return (1.0 - pow(x,2))*(1.0 - pow(y,2));
}
float f(float x,float y){
    
    return 2*(1.0 - pow(x,2)) + 2*(1.0 - pow(y,2));
}

int main(int argc,char *argv[]){
    MPI_Init(&argc,&argv);
    int rank,size;//这里使用16个进程
    
    float ela;
    float bound[2][2] = {{-1.0,1.0},{-1.0,1.0}};
    int i,j;
    float hx = (bound[0][1] - bound[0][0])/(N - 1),hy = (bound[1][1] - bound[1][0])/(N - 1);
    float x,y;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int size_x = size;
    int step_x = N/size_x;
    
    float u_1d[N][N],u_new[N][N],u[N][N];
    float f_1d[N][N];
    time0 = clock();
    //初始化所有数据
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            x = bound[0][0] + i*hx,y = bound[1][0] + j*hy;
            u[i][j] = u_acc(x,y);
            //u_1d[i][j] = 0;
            f_1d[i][j] = 0;
            u_new[i][j] = 0;
            
        }
    }
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            x = bound[0][0] + i*hx,y = bound[1][0] + j*hy;
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1){
                u_new[i][j] = u_acc(x,y);
            }
            else {
                f_1d[i][j] = f(x,y);
            }
            
            
        }
    }
    //-----------------------------------
    if (rank == 0){
        printf("the process:%d,the max_iter:%d...........\n",size,max_iter);
    }
    int k = 0;
    float right,myrho,rho,myerror,error = 0,eps = 1e-11;
    float r1 = -0.5,r2 = -pow(hx/hy,2),r3 = -pow(hy/hx,2),r = 2*(1 - r2 - r3);
    int count = 0,tag = 0;
    //MPI_Request myreq1,myreq2;
    while (k < max_iter) {
        myerror = 0;
        myrho = 0;
        rho = 0;
        error = 0;
        for(i = 0; i < N; i++){
            for(j = 0; j < N; j++){
                u_1d[i][j] = u_new[i][j];
            }
        }
        if (size > 1){
            if (rank == 0){
                count = 1;
                tag = 0;
                
                for (i = 0; i < N; i++){
                    MPI_Send(&u_1d[i][(rank + 1)*step_x - 1],count,MPI_FLOAT,rank + 1,0,MPI_COMM_WORLD);
		            MPI_Recv(&u_1d[i][(rank + 1)*step_x],count,MPI_FLOAT,rank + 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
                
                
            }
            else if (rank > 0 && rank < size - 1) {
                count = 1;
                tag = 0;
                for (i = 0; i < N; i++){
                    MPI_Recv(&u_1d[i][rank*step_x - 1],count,MPI_FLOAT,rank - 1,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    MPI_Send(&u_1d[i][rank*step_x],count,MPI_FLOAT,rank - 1,tag,MPI_COMM_WORLD);

                    MPI_Send(&u_1d[i][(rank + 1)*step_x - 1],count,MPI_FLOAT,rank + 1,0,MPI_COMM_WORLD);
		            MPI_Recv(&u_1d[i][(rank + 1)*step_x],count,MPI_FLOAT,rank + 1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
                
            }
        
            else {
                
                count = 1;
                tag = 0;
                for (i = 0; i < N; i++){
                    MPI_Recv(&u_1d[i][rank*step_x - 1],count,MPI_FLOAT,rank - 1,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    MPI_Send(&u_1d[i][rank*step_x],count,MPI_FLOAT,rank - 1,tag,MPI_COMM_WORLD);
                }
            }
        
        }
        for(i = 0; i < N; i++){
            for(j = rank*step_x; j < (rank + 1)*step_x; j++){
                x = bound[0][0] + i*hx,y = bound[1][0] + j*hy;
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1){
                    continue;
                }
                else {
                    right = (u_1d[i - 1][j + 1]*r1 + u_1d[i][j + 1]*r2 + u_1d[i + 1][j + 1]*r1
                    + u_1d[i - 1][j]*r3 + u_1d[i][j]*r + u_1d[i + 1][j]*r3 
                    + u_1d[i - 1][j - 1]*r1 + u_1d[i][j - 1]*r2 + u_1d[i + 1][j - 1]*r1 - f_1d[i][j]*(pow(hx,2) + pow(hy,2)));

                    u_new[i][j] = u_1d[i][j] - right/r;
                    myrho += right*right;
                    myerror = max(myerror,fabs(u[i][j] - u_new[i][j]));
                }
            }
        }
        if (size > 1){
            MPI_Allreduce(&myrho,&rho,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
             rho = sqrt(rho);
            MPI_Allreduce(&myerror,&error,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
        }
        else {
            rho = sqrt(myrho);
            error = myerror;
        }
        
        if (rho < eps){
            
            break;
        }
        //printf("the epoch:%d,the risid:%.2e,the err:%.2e\n",k,rho,err);
        if (rank == 0){
            if (k%1000 == 0){
                printf(".the epoch:%d,the risid:%.2e,the err:%.2e\n",k,rho,error);
                
            }
        }
        
        k += 1;

    }
    
    time1 = clock();
    ela = (float)(time1 - time0)/CLOCKS_PER_SEC;
    if (rank == 0){
        printf("Finish at %d,use time:%.2f, the resid:%.2e,the err:%.2e\n",k,ela,rho,error);
    }
    
    /**
    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++){
            printf("err[%d][%d] = %.2e\n",i,j,fabs(u[i][j] - u_new[i][j]));
        }
    }
    **/
    
    MPI_Finalize();
    
    
    return 0;
}

