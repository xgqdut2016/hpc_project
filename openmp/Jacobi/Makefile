CC = gcc
FLAGS = -O3 -Wall
OBJ = *.o
EXE = mat basic static orphaning nest nest_fun end 

all: ${EXE}

%: omp_%.c
	$(CC) -o $@ $^ $(FLAGS) -fopenmp -lm
simd_end: omp_simd_end.c
	$(CC) -o $@ $^ $(FLAGS) -fopenmp -lm -fopt-info-vec -fno-tree-vectorize
clean:
	rm -f $(OBJ) $(EXE)



