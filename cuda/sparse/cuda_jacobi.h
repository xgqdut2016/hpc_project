
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

void cuda_solve(const int *row_ptr, const int *col_ind, const double *values, const int num_rows, int num_vals, double *u_new, double *rig);

