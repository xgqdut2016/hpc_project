#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include<time.h>

#define max(a,b) ((a)>(b)?(a):(b))
#define N 128
#define max_iter 20000
#define alpha 0.5
float u_acc(float x,float y){
    return (1.0 - pow(x,2))*(1.0 - pow(y,2));
}
float f(float x,float y){
    
    return 2*(1.0 - pow(x,2)) + 2*(1.0 - pow(y,2)) + alpha*(1.0 - pow(x,2))*(1.0 - pow(y,2));
}

int main(int argc,char *argv[]){
    MPI_Init(&argc,&argv);
    int rank,size;//这里使用16个进程
    
    
    float bound[2][2] = {{-1.0,1.0},{-1.0,1.0}};
    int i,j;
    float hx = (bound[0][1] - bound[0][0])/(N - 1),hy = (bound[1][1] - bound[1][0])/(N - 1);
    float x,y;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int size_y = size;
    int step_y = N/size_y,step_x = N;
    
    float u_1d[N][N],u_new[N][N],u[N][N];
    float f_1d[N][N];
    
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
    
    int k = 0;
    float right,myrho,rho,myerror,error = 0,eps = 1e-11;
    float r1 = -0.5,r2 = -pow(hx/hy,2),r3 = -pow(hy/hx,2),r = 2*(1 - r2 - r3) + alpha*(hx*hx + hy*hy);
    int count = 0,tag = 0;
    double all_time0,all_time1,all_ela;//all time
    double commu_time0 = 0,commu_time1 = 0,commu_ela = 0,commu_t = 0,c_t = 0;//communication time
    double update_time0,update_time1,update_ela,update_t,u_t = 0;// update u_old = u_new
    double jacobi_time0,jacobi_time1,jacobi_ela,jacobi_t = 0,j_t = 0;// Jacobi time
    //double compu_t = 0;// computation time
    int up = rank - 1,down = rank + 1; // xu ni rank
    if (up < 0){
        up = MPI_PROC_NULL;
    }
    if (down == size){
        down = MPI_PROC_NULL;
    }
    
    all_time0 = MPI_Wtime();//------------------------------Jacobi start
    while (k < max_iter) {
        myerror = 0;//rank error
        myrho = 0;// rank resid
        rho = 0;//all error
        error = 0;// all resid
        //---------------------------------------------------------------------------
        update_time0 = MPI_Wtime();
        for(i = rank*step_y; i < (rank + 1)*step_y; i++){
            for(j = 0; j < N; j++){
                u_1d[i][j] = u_new[i][j];
            }
        }
        update_time1 = MPI_Wtime();
        update_ela = update_time1 - update_time0;
        
        if (size > 1){
            MPI_Allreduce(&update_ela,&update_t,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);// Barrier
        }
        else {
            update_t = update_ela;
        }
        u_t += update_t/size; // mean time
        //---------------------------------------------------------------------------
        if (size > 1){
            commu_time0 = MPI_Wtime();
            
            count = step_x;
            tag = 0;
            if (down < size){
                MPI_Sendrecv(&u_1d[(rank + 1)*step_y - 1][0],count,MPI_FLOAT,down, tag,\
                &u_1d[(rank + 1)*step_y][0],count,MPI_FLOAT,down,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            
            if (up >= 0){
                MPI_Sendrecv(&u_1d[rank*step_y][0],count,MPI_FLOAT,up, tag,\
                &u_1d[rank*step_y - 1][0],count,MPI_FLOAT,up,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            
            
            commu_time1 = MPI_Wtime();
            commu_ela = commu_time1 - commu_time0;
            MPI_Allreduce(&commu_ela,&commu_t,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);// Barrier
            c_t += commu_t/size;
        }
        //---------------------------------------------------------------------------
        jacobi_time0 = MPI_Wtime();
        for(i = rank*step_y; i < (rank + 1)*step_y; i++){
            for(j = 0; j < N; j++){
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
        
        //---------------------------------------------------------------------------
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
        jacobi_time1 = MPI_Wtime();
        jacobi_ela = jacobi_time1 - jacobi_time0;
        if (size > 1){
            MPI_Allreduce(&jacobi_ela,&jacobi_t,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);// Barrier
        }
        else {
            jacobi_t = jacobi_ela;
        }
        j_t += jacobi_t/size;
        
        k += 1;

    }
    
    all_time1 = MPI_Wtime();//---------------------Jacobi end
    all_ela = all_time1 - all_time0;
    if (rank == 0){
        printf("Use process:%d,finish at %d,use time:%.2f, the resid:%.2e,the err:%.2e\n",size,k,all_ela,rho,error);
        printf("the update time:%.2f\n",u_t);
        printf("the jacobi time:%.2f\n",j_t);
        printf("the communication time:%.2f\n",c_t);
    }
    
    MPI_Finalize();
    
    
    return 0;
}

