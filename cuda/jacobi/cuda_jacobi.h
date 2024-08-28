
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

void cuda_solve(double *f_1d, double *u_old, double *u_new, double eps, double r1, double r2, double r3, double r, int M, int N);
