
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define max_function(a,b) ((a)>(b)?(a):(b))
#define M 128
#define N 128
#define alpha 0.5
#define max_iter 20000

double u_acc(double x,double y){
    return (1.0 - pow(x,2))*(1.0 - pow(y,2));
}
double f(double x,double y){
    
    return 2*(1.0 - pow(x,2)) + 2*(1.0 - pow(y,2)) + alpha*u_acc(x,y);
}
double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}
void init_data(double bound[][2], double *u_new, double *u, double *f_1d, double dx, double dy){
    int i,j;
    double xx,yy;
    
    for(j = 0; j < N; j++){
        for(i = 0; i < M; i++){
            xx = bound[0][0] + i*dx;
            yy = bound[1][0] + j*dy;
            if (j == 0 || j == N - 1 || i == 0 || i == M - 1){
                f_1d[j*M + i] = 0;
                u_new[j*M + i] = u_acc(xx,yy);
                u[j*M + i] = u_acc(xx,yy); 
            }
            else{
                f_1d[j*M + i] = f(xx,yy)*(dx*dx + dy*dy);
                u_new[j*M + i] = 0;
                u[j*M + i] = u_acc(xx,yy);
            }
        }
    }
}

double Jacobi(double *f_1d, double *u_old, double *u_new, double r1, double r2, double r3, double r){
    int j,i;
    double error = 0;
    double resid = 0;
    for(j = 0; j < N; j++){
        for(i = 0; i < M; i++){
            u_old[j*M + i] = u_new[j*M + i];
        }
    }
    for(j = 0; j < N; j++){
        for(i = 0; i < M; i++){
            if (j == 0 || j == N - 1 || i == 0 || i == M - 1){
                continue;
            }
            else{
                resid = f_1d[j*M + i] - (r1*(u_old[(j - 1)*M + i - 1] + u_old[(j - 1)*M + i + 1]) + \
                r3*(u_old[j*M + i - 1] + u_old[j*M + i + 1]) + \
                r1*(u_old[(j + 1)*M + i - 1] + u_old[(j + 1)*M + i + 1]) + \
                r2*(u_old[(j - 1)*M + i] + u_old[(j + 1)*M + i]) + r*u_old[j*M + i]);
                u_new[j*M + i] = u_old[j*M + i] + resid/r;
                error += resid*resid;
                
            }
        }
    }
    return error;
}
void solve(double *f_1d, double *u_old, double *u_new, double eps, double r1, double r2, double r3, double r){
    int k = 0;
    double error = 0;
    int i,j;
    while (k < max_iter){
        error = Jacobi(f_1d,u_old, u_new, r1, r2, r3, r);
        if (error < eps){
            break;
        }
        k += 1;
    }
}
double L1_err(double *u_new, double *u){
    int i,j;
    double err = 0;
    for(j = 0; j < N; j++){
        for(i = 0; i < M; i++){
            err = max_function(err,fabs(u_new[j*M + i] - u[j*M + i]));
        }
    }
    return err;
}
int main(int argc, char *argv[]){
    double bound[2][2] = {{-1,1},{-1,1}};
    double dx,dy;
    dx = (bound[0][1] - bound[0][0])/ (M - 1);
    dy = (bound[1][1] - bound[1][0])/ (N - 1);

    double r1 = -0.5,r2 = -pow(dx/dy,2);
    double r3 = -pow(dy/dx,2),r = 2*(1 - r2 - r3) + alpha*(dx*dx + dy*dy);
    double u_old[M*N],u_new[M*N],f_1d[M*N],u[M*N],eps = 1e-10;
    init_data(bound, u_new, u, f_1d, dx, dy);
    double st,ela;
    st = get_walltime();
    solve(f_1d, u_old, u_new, eps, r1, r2, r3, r);
    ela = get_walltime() - st;
    double err = 0;
    err = L1_err(u_new,u);
    
    printf("Finish: use time:%.2f,the L1_err:%.4e\n",ela,err);
    
    return 0;


}

