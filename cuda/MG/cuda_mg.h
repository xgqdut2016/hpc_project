#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
void cuda_jacobi_solve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N);
void cuda_twosolve(int epoch, double eps,double bound[][2], double *u_old, double *u_new, double *b, double *resid, int M, int N);


