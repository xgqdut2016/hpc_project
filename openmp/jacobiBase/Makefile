CC = gcc
FLAGS = -O3 -Wall
OBJ = *.o
EXE = basic basic3 five

all: ${EXE}

basic: omp_basic.c
        $(CC) -o $@ $^ $(FLAGS) -fopenmp -lm
basic3: omp_basic3.c
        $(CC) -o $@ $^ $(FLAGS) -fopenmp -lm
five: omp_five.c
        $(CC) -o $@ $^ $(FLAGS) -fopenmp -lm

clean:
        rm -f $(OBJ) $(EXE)


