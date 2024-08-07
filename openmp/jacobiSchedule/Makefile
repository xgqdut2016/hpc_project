CC = gcc
FLAGS = -O3 -Wall
OBJ = *.o
EXE = schedule static runtime dynamic

all: ${EXE}
schedule: omp_schedule.c
        $(CC) -o $@ $^ $(FLAGS) -fopenmp -lm
static: omp_static.c
        $(CC) -o $@ $^ $(FLAGS) -fopenmp -lm
runtime: omp_runtime.c
        $(CC) -o $@ $^ $(FLAGS) -fopenmp -lm
dynamic: omp_dynamic.c
        $(CC) -o $@ $^ $(FLAGS) -fopenmp -lm

clean:
        rm -f $(OBJ) $(EXE)

