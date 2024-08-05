#include "mpi.h"
#include <stdio.h>
#include <math.h>

#define PI25DT 3.141592653589793238462643
#define REPEAT 20

int main(int argc, char *argv[]){
  int    size, rank, n, i, k;
  double mypi, pi, h, sum, x, t0, t1;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) n = 10000000;
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  t0 = MPI_Wtime();

  for (k = 0; k < REPEAT; k++) {
    h = 1.0 / (double) n;
    sum = 0.0;
    for (i = rank + 1; i <= n; i += size) {
      x = h * ((double)i - 0.5);
      sum += sqrt(1.-x*x);
    }
    mypi = 4.0 * h * sum;
    MPI_Allreduce(&mypi, &pi, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  t1 = MPI_Wtime();

  if (rank == 0) {
    printf("Number of processes = %d\n", size);
    printf(" pi is approximately %.16f\n", pi);
    printf(" Error is %.16f\n", fabs(pi-PI25DT));
    printf(" Wall clock time = %f\n", (t1-t0)/REPEAT);
  }

  MPI_Finalize();
  return 0;
}


