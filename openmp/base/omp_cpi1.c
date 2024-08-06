#include <omp.h>
#include <stdio.h>
#include <math.h>

#define PI25DT 3.141592653589793238462643

int main(int argc, char *argv[]){
  int    nthreads, tid, n, i;
  double pi, h, x, t0, t1;

  t0 = omp_get_wtime();
  n = 10000000;
  h = 1.0 / (double) n;
  pi = 0.0;
#pragma omp parallel default(none) \
  private(tid, i, x) reduction(+:pi) shared(h,n,threads)
  {
    nthreads = omp_get_num_threads();
    tid = omp_get_thread_num();
    for (i = tid + 1; i <= n; i += nthreads) {
      x = h * ((double)i - 0.5);
      pi += 4.0 * h * sqrt(1.-x*x);
    }
  }
  t1 = omp_get_wtime();
  printf(" Number of threads = %d\n", nthreads);
  printf(" pi is approximately %.16f\n", pi);
  printf(" Error is %.16f\n", fabs(pi-PI25DT));
  printf(" Wall clock time = %f\n", t1-t0);

  return 0;
}


