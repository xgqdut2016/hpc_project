#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include "cuda_mg.h"

#define max_function(a,b) ((a)>(b)?(a):(b))

#define gepoch ((int)10)
double UU(int prob, int *order, double x, double y){
    if (prob == 1){
        double temp = 10*pow(x + y,2) + pow(x - y,2) + 0.5;
        if (order[0] == 0 && order[1] == 0){
            return log(temp);
        }
        if (order[0] == 2 && order[1] == 0){
            return - pow(temp,-2)*pow(20*(x + y) + 2*(x - y),2) \
                   + 22*pow(temp,-1);
        }
        if (order[0] == 0 && order[1] == 2){
            return - pow(temp,-2)*pow(20*(x + y) - 2*(x - y),2) \
                   + 22*pow(temp,-1);
        }
    }
    if (prob == 2){
        if (order[0] == 0 && order[1] == 0){
            return 0.5*(pow(x,3) - x)*(exp(2*y) + exp(-2*y));
        }
        if (order[0] == 2 && order[1] == 0){
            return 3*x*(exp(2*y) + exp(-2*y));
        }
        if (order[0] == 0 && order[1] == 2){
            return 2*(pow(x,3) - x)*(exp(2*y) + exp(-2*y));
        }
    }
    if (prob == 3){
        double temp1 = pow(x,2) - pow(y,2);
        double temp2 = pow(x,2) + pow(y,2) + 0.1;
        if (order[0] == 0 && order[1] == 0){
            return temp1/temp2;
        }
        if (order[0] == 2 && order[1] == 0){
            return (2)*pow(temp2,-1) + \
                   2*(2*x)*(-1)*pow(temp2,-2)*(2*x) + \
                   temp1*(2)*pow(temp2,-3)*pow(2*x,2) + \
                   temp1*(-1)*pow(temp2,-2)*(2);
        }
        if (order[0] == 0 && order[1] == 2){
            return (-2)*pow(temp2,-1) + \
                   2*(-2*y)*(-1)*pow(temp2,-2)*(2*y) + \
                   temp1*(2)*pow(temp2,-3)*pow(2*y,2) + \
                   temp1*(-1)*pow(temp2,-2)*(2);

        }
    }
}

double FF(int prob, double x, double y){
    int order_x[2] = {2,0};
    int order_y[2] = {0,2};
    return -UU(prob,order_x,x,y) - UU(prob,order_y,x,y);
}
double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6); 
}
void init_data(int prob, double bound[][2], double *u_acc, double *b, int M, int N){
    int i,j;
    int order[2] = {0,0};
    double dx = (bound[0][1] - bound[0][0])/M;
    double dy = (bound[1][1] - bound[1][0])/N;
    double xx,yy;
    
    for(j = 0; j < N + 1; j++){
        for(i = 0; i < M + 1; i++){
            xx = bound[0][0] + i*dx;
            yy = bound[1][0] + j*dy;
            u_acc[j*(M + 1) + i] = UU(prob,order,xx,yy);
            if (j == 0 || j == N || i == 0 || i == M){
                b[j*(M + 1) + i] = UU(prob,order,xx,yy);
            }
            else{
                b[j*(M + 1) + i] = FF(prob,xx,yy)*dx*dy;
            }
            
        }
    }
}

double Gauss(double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N){
    int i,j;
    double dx = (bound[0][1] - bound[0][0])/M;
    double dy = (bound[1][1] - bound[1][0])/N;
    double r1 = dy/dx, r2 = dx/dy;
    double r = 2*(r1 + r2);
    double temp;
    double error = 0;
    
    for(j = 0; j < N + 1; j++){
        for(i = 0; i < M + 1; i++){
            if (j == 0 || j == N || i == 0 || i == M){
                u_new[j*(M + 1) + i] = b[j*(M + 1) + i];
            }
            else{
                temp = b[j*(M + 1) + i] + \
                r1*(u_new[j*(M + 1) + i - 1] + u_old[j*(M + 1) + i + 1]) + \
                r2*(u_new[(j - 1)*(M + 1) + i] + u_old[(j + 1)*(M + 1) + i]) - \
                r*u_old[j*(M + 1) + i];
                u_new[j*(M + 1) + i] = u_old[j*(M + 1) + i] + temp/r;
                resid[j*(M + 1) + i] = temp;
                error += temp*temp;
            }
        }
    }
    
    return sqrt(error/((M + 1)*(N + 1)));

}
double Jacobi(double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N){
    int i,j;
    double dx = (bound[0][1] - bound[0][0])/M;
    double dy = (bound[1][1] - bound[1][0])/N;
    double r1 = dy/dx, r2 = dx/dy;
    double r = 2*(r1 + r2);
    double temp;
    double error = 0;
    
    for(j = 0; j < N + 1; j++){
        for(i = 0; i < M + 1; i++){
            if (j == 0 || j == N || i == 0 || i == M){
                u_new[j*(M + 1) + i] = b[j*(M + 1) + i];
            }
            else{
                temp = b[j*(M + 1) + i] + \
                r1*(u_old[j*(M + 1) + i - 1] + u_old[j*(M + 1) + i + 1]) + \
                r2*(u_old[(j - 1)*(M + 1) + i] + u_old[(j + 1)*(M + 1) + i]) - \
                r*u_old[j*(M + 1) + i];
                u_new[j*(M + 1) + i] = u_old[j*(M + 1) + i] + temp/r;
                resid[j*(M + 1) + i] = temp;
                error += temp*temp;
            }
        }
    }
    
    return sqrt(error/((M + 1)*(N + 1)));

}
void copy(double *u_new, double *u_old, int M, int N){
    for(int j = 0; j < N + 1; j++){
        for(int i = 0; i < M + 1; i++){
            u_old[j*(M + 1) + i] = u_new[j*(M + 1) + i];
        }
    }
}
double Gauss_solve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N){
    int k = 0;
    double error = 0;
    while(k < epoch){
        error = Gauss(bound, u_old, u_new, b, resid, M, N);
        if(error < eps){
            break;
        }
        else{
            copy(u_new, u_old, M, N);
            k += 1;
        }
    }
    return error;
}
double Jacobi_solve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N){
    int k = 0;
    double error = 0;
    while(k < epoch){
        error = Jacobi(bound, u_old, u_new, b, resid, M, N);
        if(error < eps){
            break;
        }
        else{
            copy(u_new, u_old, M, N);
            k += 1;
        }
    }
    return error;
}

double L2_norm(double *u, int M, int N){
    double error = 0;
    for(int j = 0; j < N + 1; j++){
        for(int i = 0; i < M + 1; i++){
            error += pow(u[j*(M + 1) + i],2);
        }
    }
    return sqrt(error/((M + 1)*(N + 1)));
}
double L1_err(double *u_new, double *u, int M, int N){
    int i,j;
    double err = 0;
    for(j = 0; j < N + 1; j++){
        for(i = 0; i < M + 1; i++){
            err = max_function(err,fabs(u_new[j*(M + 1) + i] - u[j*(M + 1) + i]));
        }
    }
    return err;
}
void init0(double *u, int M, int N){
    for(int j = 0; j < N + 1; j ++){
        for(int i = 0; i < M + 1; i++){
            u[j*(M + 1) + i] = 0;
        }
    }
}
void low(int M, int N, double *u_long, double *u_short){
    int m = M/2;
    int n = N/2;
    //----------------
    u_short[0] = u_long[0];
    u_short[m] = u_long[M];
    u_short[n*(m + 1)] = u_long[N*(M + 1)];
    u_short[n*(m + 1) + m] = u_long[N*(M + 1) + M];
    //--------------------
    for(int j = 1; j < n; j++){
        u_short[j*(m + 1)] = 0.5*(u_long[(2*j - 1)*(M + 1)] + u_long[(2*j + 1)*(M + 1)]);
        u_short[j*(m + 1) + m] = 0.5*(u_long[(2*j - 1)*(M + 1) + M] + u_long[(2*j + 1)*(M + 1) + M]);
    }
    for(int i = 1; i < m; i++){
        u_short[i] = 0.5*(u_long[2*i - 1] + u_long[2*i + 1]);
        u_short[n*(m + 1) + i] = 0.5*(u_long[N*(M + 1) + 2*i - 1] + u_long[N*(M + 1) + 2*i + 1]);
    }
    double temp = 0;
    for(int j = 1; j < n; j++){
        for(int i = 1; i < m; i++){
            temp = u_long[(2*j - 1)*(M + 1) + 2*i - 1] + u_long[(2*j - 1)*(M + 1) + 2*i + 1] + \
            u_long[(2*j + 1)*(M + 1) + 2*i - 1] + u_long[(2*j + 1)*(M + 1) + 2*i + 1] + \
            2*(u_long[(2*j - 1)*(M + 1) + 2*i] + u_long[(2*j + 1)*(M + 1) + 2*i]) + \
            2*(u_long[2*j*(M + 1) + 2*i - 1] + u_long[2*j*(M + 1) + 2*i + 1]) + \
            4*u_long[2*j*(M + 1) + 2*i];
            u_short[j*(m + 1) + i] = temp/16.0;
        }
    }
}

void high(int m, int n, double *u_short, double *u_long){
    int M = 2*m;
    int N = 2*n;
    for(int j = 0; j < n; j++){
        for(int i = 0; i < m; i++){
            u_long[2*j*(M + 1) + 2*i] = u_short[j*(m + 1) + i];
            u_long[(2*j + 1)*(M + 1) + 2*i] = 0.5*(u_short[j*(m + 1) + i] + u_short[(j + 1)*(m + 1) + i]);
            u_long[2*j*(M + 1) + 2*i + 1] = 0.5*(u_short[j*(m + 1) + i] + u_short[j*(m + 1) + i + 1]);
            u_long[(2*j + 1)*(M + 1) + 2*i + 1] = 0.25*\
            (u_short[(j + 1)*(m + 1) + i] + u_short[(j + 1)*(m + 1) + i + 1] + u_short[j*(m + 1) + i + 1] + u_short[j*(m + 1) + i]);
        }
    }
}
void plus(double *y, double *x, double a, int M, int N){
    for(int j = 0; j < N + 1; j++){
        for(int i = 0; i < M + 1; i++){
            y[j*(M + 1) + i] = y[j*(M + 1) + i] + a*x[j*(M + 1) + i];
        }
    }
}

double twosolve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N){
    int k = 0;
    double error = 0;
    //------------
    int m = M/2,n = N/2;
    int sizesmall = (m + 1)*(n + 1)*sizeof(double);
    double *b_short, *u_short, *u_short_old, *resid_short;

    b_short = (double *)malloc(sizesmall);
    u_short = (double *)malloc(sizesmall);
    u_short_old = (double *)malloc(sizesmall);
    resid_short = (double *)malloc(sizesmall);
    
    while(k < epoch){
        //error = Gauss_solve(gepoch, eps, bound, u_old, u_new, b, resid, M, N);
        error = Jacobi_solve(gepoch, eps, bound, u_old, u_new, b, resid, M, N);
        
        low(M, N, resid, b_short);
        init0(u_short_old, m, n);
        //error = Gauss_solve(gepoch, eps, bound, u_short_old, u_short, b_short, resid_short, m,n);
        error = Jacobi_solve(gepoch, eps, bound, u_short_old, u_short, b_short, resid_short, m,n);
        init0(u_old,M,N);high(m, n, u_short, u_old);
        plus(u_new, u_old, 1.0, M, N);
        if(error < eps){
            break;
        }
        else{
            copy(u_new, u_old, M, N);
            k += 1;
        }
    }
    free(u_short);
    free(b_short);
    free(u_short_old);
    free(resid_short);
    
    return error;
}
void Vcycle(double eps, double bound[][2], double *u_new, double *resid, int M, int N, int count){
    double *rig[count + 1];
    double *res[count + 1];
    double *sol_old[count + 1];
    double *sol[count + 1];
    double error;
    int m = 2*M, n = 2*N, size;
    
    for(int k = 0; k < count + 1; k++){
        m = m/2;
        n = n/2;
        size = (m + 1)*(n + 1)*sizeof(double);
        rig[k] = (double *)malloc(size),init0(rig[k], m, n);
        res[k] = (double *)malloc(size),init0(res[k], m, n);
        sol_old[k] = (double *)malloc(size),init0(sol_old[k], m, n);
        sol[k] = (double *)malloc(size),init0(sol[k], m, n);
    }
    
    copy(resid, res[0], M, N);
    
    m = M, n = N;
    for(int k = 0; k < count; k++){
        m = m/2;
        n = n/2;
        low(m, n, res[k], rig[k + 1]);
        init0(sol_old[k + 1], m, n);
        error = Gauss_solve(gepoch, eps, bound, sol_old[k + 1], sol[k + 1], rig[k + 1], res[k + 1], m,n);
    }
    
    for(int k = count; k > 0; k--){
        m = m*2;
        n = n*2;
        init0(sol_old[k - 1], m, n);
        high(m/2, n/2, sol[k], sol_old[k - 1]);
        plus(sol[k - 1], sol_old[k - 1], 1.0, m, n);
        copy(sol[k - 1], sol_old[k - 1], m , n);
        error = Gauss_solve(gepoch, eps, bound, sol_old[k - 1], sol[k - 1], rig[k - 1], res[k - 1], m,n);
    }
    
    plus(u_new, sol[0], 1.0, M, N);
    
}
double Vsolve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N, int count){
    int k = 0;
    
    double error = 0;
    while (k < epoch){
        error = Gauss_solve(gepoch, eps, bound, u_old, u_new, b, resid, M, N);
        init0(u_old,M,N);Vcycle(eps, bound, u_old, resid, M, N, count);
        plus(u_new, u_old, 1.0, M, N);
        if(error < eps){
            break;
        }
        else{
            copy(u_new, u_old, M, N);
            k += 1;
        }
    }
    return error;
}
int main(int argc, char **argv){
    double bound[2][2] = {{-1,1},{-1,1}};
    double dx,dy;
    int M = 128;  
    int N = 64;
    int prob = 1;
    int epoch = 2000;
    double eps = 1e-10;
    double err = 0;
    double st,ela;
    double Gserr;
    if (argc > 1) prob = atoi(argv[1]); // user-specified value
    printf("problem:%d, grid = (%d,%d), max_iter = %d,--------------------\n",prob,M,N,epoch);
    int size = (M + 1)*(N + 1)*sizeof(double);
    double *u_old,*u_new,*u_acc,*b, *resid;
    u_old = (double *)malloc(size);
    u_new = (double *)malloc(size);
    u_acc = (double *)malloc(size);
    b = (double *)malloc(size);
    resid = (double *)malloc(size);
    //-----------Jacobi
    init0(u_old, M, N),init0(u_new, M, N),init0(b, M, N),init0(resid, M, N);
    init_data(prob, bound, u_acc, b, M, N);
    
    st = get_walltime();
    Gserr = Jacobi_solve(epoch, eps, bound, u_old, u_new, b, resid, M, N);
    ela = get_walltime() - st;
    err = L1_err(u_new,u_acc,M,N);
    printf("Jacobi iter error:%.3e,error:%.3e, use time:%.2f\n", Gserr, err, ela);
    //----------Gauss
    init0(u_old, M, N),init0(u_new, M, N),init0(b, M, N),init0(resid, M, N);
    init_data(prob, bound, u_acc, b, M, N);
    
    st = get_walltime();
    Gserr = Gauss_solve(epoch, eps, bound, u_old, u_new, b, resid, M, N);
    ela = get_walltime() - st;
    err = L1_err(u_new,u_acc,M,N);
    printf("Gauss iter error:%.3e,error:%.3e, use time:%.2f\n", Gserr, err, ela);
    //--------------------------twosolve
    
    init0(u_old, M, N),init0(u_new, M, N),init0(b, M, N),init0(resid, M, N);
    init_data(prob, bound, u_acc, b, M, N);
    st = get_walltime();
    Gserr = twosolve(epoch, eps, bound, u_old, u_new, b, resid, M, N);
    ela = get_walltime() - st;
    err = L1_err(u_new,u_acc,M,N);
    printf("Twosolve iter error:%.3e,error:%.3e, use time:%.2f\n", Gserr, err, ela);
    //--------------------------Vsolve
    int count = 1;
    init0(u_old, M, N),init0(u_new, M, N),init0(b, M, N),init0(resid, M, N);
    init_data(prob, bound, u_acc, b, M, N);
    st = get_walltime();
    Gserr = Vsolve(epoch, eps, bound, u_old, u_new, b, resid, M, N, count);
    ela = get_walltime() - st;
    err = L1_err(u_new,u_acc,M,N);
    printf("Vsolve iter error:%.3e,error:%.3e, use time:%.2f\n", Gserr, err, ela);
    //---------------parallel jacobi
    init0(u_old, M, N),init0(u_new, M, N),init0(b, M, N),init0(resid, M, N);
    init_data(prob, bound, u_acc, b, M, N);
    st = get_walltime();
    
    cuda_jacobi_solve(epoch, eps, bound, u_old, u_new, b, resid, M, N);
    ela = get_walltime() - st;
    err = L1_err(u_new,u_acc,M,N);
    printf("parallel Jacobi error:%.3e, use time:%.2f\n", err, ela);
    //---------------parallel Twosolve
    init0(u_old, M, N),init0(u_new, M, N),init0(b, M, N),init0(resid, M, N);
    init_data(prob, bound, u_acc, b, M, N);
    st = get_walltime();
    
    cuda_twosolve(epoch, eps, bound, u_old, u_new, b, resid, M, N);
    ela = get_walltime() - st;
    err = L1_err(u_new,u_acc,M,N);
    printf("parallel Twosolve error:%.3e, use time:%.2f\n", err, ela);
    free(u_old);
    free(u_new);
    free(u_acc);
    free(b);
    free(resid);

    return 0;


}



