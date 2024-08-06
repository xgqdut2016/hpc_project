CC = gcc -std=c99
FLAGS = -O3 -Wall -lm
OBJ = *.o
EXE = v1 v2 v3 v4 v5 

all: ${EXE}

%: omp_%.c
	$(CC) -o $@ $^ $(FLAGS) -fopenmp -fopt-info-vec -fno-tree-vectorize

clean:
	rm -f $(OBJ) $(EXE)




