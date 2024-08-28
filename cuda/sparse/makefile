CC = gcc
CFLAGS = -std=c99

NVCC = nvcc
NVCC_FLAGS = -std=c++11 -O3 -Wno-deprecated-gpu-targets

LIBRARIES = -L/${CUDA_DIR}/lib64 -lcudart -lm

EXE = jacobi
all: ${EXE}

jacobi: main.o cuda_jacobi.o
		$(CC) $^ -o $@ $(LIBRARIES)

main.o: main.c
		$(CC) $(CFLAGS) -c $^ -o $@

cuda_jacobi.o: cuda_jacobi.cu
		$(NVCC) $(NVCC_FLAGS) -c $^ -o $@
create: matvec_create.c
		
		$(CC) -o $@ $^ $(CFLAGS) -lm
		./create
clean:
		rm *.vec
		rm *.mtx
		rm *.o 
		rm create
		rm jacobi




