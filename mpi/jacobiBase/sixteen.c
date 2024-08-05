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
    int size_x = 4,size_y = 4;//s_x,s_y分别表示不同进程的分割
    int step_x = N/size_x,step_y = N/size_y;
    float ela;
    float bound[2][2] = {{-1.0,1.0},{-1.0,1.0}};
    int i,j;
    float hx = (bound[0][1] - bound[0][0])/(N - 1),hy = (bound[1][1] - bound[1][0])/(N - 1);
    float x,y;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    
    float u_1d[N][N],u_new[N][N],u[N][N];
    float f_1d[N][N];
    time0 = clock();
    //初始化所有数据
    for(i = (rank/size_x)*step_y; i < (rank/size_x + 1)*step_y; i++){
        for(j = (rank%size_x)*step_x; j < (rank%size_x + 1)*step_x; j++){
            x = bound[0][0] + i*hx,y = bound[1][0] + j*hy;
            u[i][j] = u_acc(x,y);
            u_1d[i][j] = 0;
            f_1d[i][j] = 0;
            u_new[i][j] = 0;
            
        }
    }
    for(i = (rank/size_x)*step_y; i < (rank/size_x + 1)*step_y; i++){
        for(j = (rank%size_x)*step_x; j < (rank%size_x + 1)*step_x; j++){
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
    
    int k = 0;
    float right,myrho,rho,myerror,error = 0,eps = 1e-11;
    float r1 = -0.5,r2 = -pow(hx/hy,2),r3 = -pow(hy/hx,2),r = 2*(1 - r2 - r3);
    int count = 0,tag = 0;
    
    while (k < max_iter) {

        for(i = (rank/size_x)*step_y; i < (rank/size_x + 1)*step_y; i++){
            for(j = (rank%size_x)*step_x; j < (rank%size_x + 1)*step_x; j++){
                u_1d[i][j] = u_new[i][j];
            }
        }
        //行交换
        if (rank < 4){
            count = step_x;
            tag = 0;
            MPI_Send(&u_1d[(rank/size_x + 1)*step_y - 1][(rank%size_x)*step_x],count,MPI_FLOAT,rank + 4,tag,MPI_COMM_WORLD);
            MPI_Recv(&u_1d[(rank/size_x + 1)*step_y][(rank%size_x)*step_x],count,MPI_FLOAT,rank + 4,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        else if (rank >= 4 && rank < 12){
            count = step_x;
            tag = 0;
            MPI_Recv(&u_1d[(rank/size_x)*step_y - 1][(rank%size_x)*step_x],count,MPI_FLOAT,rank - 4,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Send(&u_1d[(rank/size_x)*step_y][(rank%size_x)*step_x],count,MPI_FLOAT,rank - 4,tag,MPI_COMM_WORLD);
            count = step_x;
            tag = 0;
            MPI_Send(&u_1d[(rank/size_x + 1)*step_y - 1][(rank%size_x)*step_x],count,MPI_FLOAT,rank + 4,tag,MPI_COMM_WORLD);
            MPI_Recv(&u_1d[(rank/size_x + 1)*step_y][(rank%size_x)*step_x],count,MPI_FLOAT,rank + 4,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        else {
            count = step_x;
            tag = 0;
            MPI_Recv(&u_1d[(rank/size_x)*step_y - 1][(rank%size_x)*step_x],count,MPI_FLOAT,rank - 4,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Send(&u_1d[(rank/size_x)*step_y][(rank%size_x)*step_x],count,MPI_FLOAT,rank - 4,tag,MPI_COMM_WORLD);
        }
        //列交换
        if (rank%4 == 0){
            count = 1;
            tag = 0;
            for(i = (rank/size_x)*step_y; i < (rank/size_x + 1)*step_y; i++){
                MPI_Send(&u_1d[i][(rank%size_x + 1)*step_x - 1],count,MPI_FLOAT,rank + 1,tag,MPI_COMM_WORLD);
                MPI_Recv(&u_1d[i][(rank%size_x + 1)*step_x],count,MPI_FLOAT,rank + 1,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
        else if (rank%4 == 1 || rank%4 == 2){
            count = 1;
            tag = 0;
            for(i = (rank/size_x)*step_y; i < (rank/size_x + 1)*step_y; i++){
                MPI_Recv(&u_1d[i][(rank%size_x)*step_x - 1],count,MPI_FLOAT,rank - 1,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Send(&u_1d[i][(rank%size_x)*step_x],count,MPI_FLOAT,rank - 1,tag,MPI_COMM_WORLD);
            }
            count = 1;
            tag = 0;
            for(i = (rank/size_x)*step_y; i < (rank/size_x + 1)*step_y; i++){
                MPI_Send(&u_1d[i][(rank%size_x + 1)*step_x - 1],count,MPI_FLOAT,rank + 1,tag,MPI_COMM_WORLD);
                MPI_Recv(&u_1d[i][(rank%size_x + 1)*step_x],count,MPI_FLOAT,rank + 1,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    
            }
        }
        else {
            count = 1;
            tag = 0;
            for(i = (rank/size_x)*step_y; i < (rank/size_x + 1)*step_y; i++){
                MPI_Recv(&u_1d[i][(rank%size_x)*step_x - 1],count,MPI_FLOAT,rank - 1,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Send(&u_1d[i][(rank%size_x)*step_x],count,MPI_FLOAT,rank - 1,tag,MPI_COMM_WORLD);
            }
        }
        //角点交换
        if (rank%size_x != 3){
            if (rank < 11){
                count = 1;
                tag = 0;
                MPI_Send(&u_1d[(rank/size_x + 1)*step_y - 1][(rank%size_x + 1)*step_x - 1],count,MPI_FLOAT,rank + 5,tag,MPI_COMM_WORLD);
                MPI_Recv(&u_1d[(rank/size_x + 1)*step_y][(rank%size_x + 1)*step_x],count,MPI_FLOAT,rank + 5,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            if (rank > 3){
                count = 1;
                tag = 0;
                MPI_Send(&u_1d[(rank/size_x)*step_y][(rank%size_x + 1)*step_x - 1],count,MPI_FLOAT,rank - 3,tag,MPI_COMM_WORLD);
                MPI_Recv(&u_1d[(rank/size_x)*step_y - 1][(rank%size_x + 1)*step_x],count,MPI_FLOAT,rank - 3,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
        if (rank%size_x != 0){
            if (rank > 3){
                count = 1;
                tag = 0;
                MPI_Recv(&u_1d[(rank/size_x)*step_y - 1][(rank%size_x)*step_x - 1],count,MPI_FLOAT,rank - 5,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Send(&u_1d[(rank/size_x)*step_y][(rank%size_x)*step_x],count,MPI_FLOAT,rank - 5,tag,MPI_COMM_WORLD);
                
            }
            if (rank < 12){
                count = 1;
                tag = 0;
                MPI_Recv(&u_1d[(rank/size_x + 1)*step_y][(rank%size_x)*step_x - 1],count,MPI_FLOAT,rank + 3,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Send(&u_1d[(rank/size_x + 1)*step_y - 1][(rank%size_x)*step_x],count,MPI_FLOAT,rank + 3,tag,MPI_COMM_WORLD);
                
            }
        }
        myerror = 0;
        myrho = 0;
        rho = 0;
        error = 0;
        for(i = (rank/size_x)*step_y; i < (rank/size_x + 1)*step_y; i++){
            for(j = (rank%size_x)*step_x; j < (rank%size_x + 1)*step_x; j++){
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
        
        MPI_Allreduce(&myrho,&rho,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
        rho = sqrt(rho);
        MPI_Allreduce(&myerror,&error,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
        if (rho < eps){
            break;
        }
        //printf("the epoch:%d,the risid:%.2e,the err:%.2e\n",k,rho,error);
        if (rank == 0){
            if (k%10 == 0){
                printf("the epoch:%d,the risid:%.2e,the err:%.2e\n",k,rho,error);
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

