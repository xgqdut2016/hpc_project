CC = mpicc
FLAGS = -O3 -Wall
OBJ = *.o
EXE = hello pingpong

all: ${EXE}

hello: mpi_hello.c
        $(CC) -o $@ $^ $(FLAGS)
pingpong: mpi_pingpong.c
        $(CC) -o $@ $^ $(FLAGS)
clean:
        rm -f $(OBJ) $(EXE)



