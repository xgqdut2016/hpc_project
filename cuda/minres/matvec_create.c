#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#define max_function(a,b) ((a)>(b)?(a):(b))
#define alpha 0.5
double u_acc(double x,double y){
    return (1.0 - pow(x,2))*(1.0 - pow(y,2));
}
double f(double x,double y){
    
    return 2*(1.0 - pow(x,2)) + 2*(1.0 - pow(y,2)) + alpha*u_acc(x,y);
}
void init_rig(double *f_1d, double *u, double bound[][2], int M, int N){
    double dx,dy,xx,yy;
    dx = (bound[0][1] - bound[0][0])/ (M - 1);
    dy = (bound[1][1] - bound[1][0])/ (N - 1);
    double r1 = -0.5,r2 = -pow(dx/dy,2);
    double r3 = -pow(dy/dx,2),r = 2*(1 - r2 - r3) + alpha*(dx*dx + dy*dy);
    int i,j;
    for(j = 0; j < N; j++){
        for(i = 0; i < M; i++){
            xx = bound[0][0] + i*dx;
            yy = bound[1][0] + j*dy;
            u[j*M + i] = u_acc(xx,yy);
            if (j == 0 || j == N - 1 || i == 0 || i == M - 1){
                f_1d[j*M + i] = u_acc(xx,yy);
                
            }
            else{
                f_1d[j*M + i] = f(xx,yy)*(dx*dx + dy*dy);
                
            }
        }
    }

}
void res(double *f_1d, double *u, double bound[][2], int M, int N){
    double dx,dy,xx,yy;
    double resid = 0;
    double err = 0;
    dx = (bound[0][1] - bound[0][0])/ (M - 1);
    dy = (bound[1][1] - bound[1][0])/ (N - 1);
    double r1 = -0.5,r2 = -pow(dx/dy,2);
    double r3 = -pow(dy/dx,2),r = 2*(1 - r2 - r3) + alpha*(dx*dx + dy*dy);
    int i,j;
    for(j = 0; j < N; j++){
        for(i = 0; i < M; i++){
            xx = bound[0][0] + i*dx;
            yy = bound[1][0] + j*dy;
            u[j*M + i] = u_acc(xx,yy);
            if (j == 0 || j == N - 1 || i == 0 || i == M - 1){
                resid = f_1d[j*M + i] - u[j*M + i];
            }
            else{
                resid = f_1d[j*M + i] - (r1*(u[(j - 1)*M + i - 1] + u[(j - 1)*M + i + 1]) + \
                r3*(u[j*M + i - 1] + u[j*M + i + 1]) + \
                r1*(u[(j + 1)*M + i - 1] + u[(j + 1)*M + i + 1]) + \
                r2*(u[(j - 1)*M + i] + u[(j + 1)*M + i]) + r*u[j*M + i]);
            }
        }
        err = max_function(err,resid);
    }
    printf("Ax = b: resid:%.4e\n",err);
}
int main(int argc, char **argv){
    double bound[2][2] = {{-1,1},{-1,1}};
    
    int M = 64;  // default value
    int N = 128;
    if (argc > 1) M = atoi(argv[1]); // user-specified value
    double f_1d[M*N],u[M*N];
    init_rig(f_1d, u, bound, M, N);

    double dx,dy,xx,yy;
    dx = (bound[0][1] - bound[0][0])/ (M - 1);
    dy = (bound[1][1] - bound[1][0])/ (N - 1);
    double r1 = -0.5,r2 = -pow(dx/dy,2);
    double r3 = -pow(dy/dx,2),r = 2*(1 - r2 - r3) + alpha*(dx*dx + dy*dy);
    int i,j;
    //--------------------------------------------------------
    const char* file_mat = "./matrix.mtx";
    FILE *fptr_mat = fopen(file_mat,"w");
    if (fptr_mat == NULL){
        printf("fail to create matrix\n");
        exit(0);
    }
    int count = M*N + 8*(M - 2)*(N - 2);
    int t = 0;
    fprintf(fptr_mat,"%d %d %d\n",M*N,M*N,count);// index + 1
    for(j = 0; j < N; j++){
        for(i = 0; i < M; i++){
            if (j == 0 || j == N - 1 || i == 0 || i == M - 1){
                fprintf(fptr_mat,"%d %d %lfe\n",j*M + i + 1, j*M + i + 1, 1.0);
                //count -= 1;
                t += 1;
            }
            else {
                fprintf(fptr_mat,"%d %d %lf\n",j*M + i + 1, (j - 1)*M + i - 1 + 1, r1);
                fprintf(fptr_mat,"%d %d %lf\n",j*M + i + 1, (j - 1)*M + i + 1 + 1, r1);
                fprintf(fptr_mat,"%d %d %lf\n",j*M + i + 1, (j - 1)*M + i + 1, r2);
                fprintf(fptr_mat,"%d %d %lf\n",j*M + i + 1, j*M + i - 1 + 1, r3);
                fprintf(fptr_mat,"%d %d %lf\n",j*M + i + 1, j*M + i + 1 + 1, r3);
                fprintf(fptr_mat,"%d %d %lf\n",j*M + i + 1, j*M + i + 1, r);
                fprintf(fptr_mat,"%d %d %lf\n",j*M + i + 1, (j + 1)*M + i - 1 + 1, r1);
                fprintf(fptr_mat,"%d %d %lf\n",j*M + i + 1, (j + 1)*M + i + 1 + 1, r1);
                fprintf(fptr_mat,"%d %d %lf\n",j*M + i + 1, (j + 1)*M + i + 1, r2);
                //count -= 9;
                t += 9;
            }
        }
    }
    fclose(fptr_mat);
    printf("non-zero number error,count:%d,t:%d\n",count,t);
    
    //----------------------------------------------
    const char* file_rig = "./rig.vec";
    FILE *fptr_rig = fopen(file_rig,"w");
    if (fptr_rig == NULL){
        printf("fail to create vector\n");
        exit(0);
    }
    fprintf(fptr_rig,"%d %d %d\n",M,N,M*N);
    for(i = 0; i < M*N; i++){
        fprintf(fptr_rig,"%lf\n",f_1d[i]);
    }
    fclose(fptr_rig);
    //--------------------------------------------
    const char* file_u = "./uacc.vec";
    FILE *fptr_u = fopen(file_u,"w");
    if (fptr_u == NULL){
        printf("fail to create vector\n");
        exit(0);
    }
    fprintf(fptr_u,"%d %d %d\n",M,N,M*N);
    for(i = 0; i < M*N; i++){
        fprintf(fptr_u,"%lf\n",u[i]);
    }
    fclose(fptr_u);
    
    res(f_1d, u, bound, M, N);
    
    return 0;


}


