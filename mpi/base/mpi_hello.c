#include <mpi.h>
#include <stdio.h>
int main(int argc,char** argv){
    int size;
    int rank;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    printf("the process of %d of %d:hello world\n",rank,size);
    if (rank == 0){
    	printf("that is all\n");
    }
    MPI_Finalize();
    return 0;
}


