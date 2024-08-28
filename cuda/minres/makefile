CC = gcc
CFLAGS = -std=c99

NVCC = nvcc
NVCC_FLAGS = -std=c++11 -O3 -Wno-deprecated-gpu-targets

LIBRARIES = -L/${CUDA_DIR}/lib64 -lcudart -lm

default: minres

minres: main.o cuda_minres.o
		$(CC) $^ -o $@ $(LIBRARIES)

main.o: main.c
		$(CC) $(CFLAGS) -c $^ -o $@

cuda_minres.o: cuda_minres.cu
		$(NVCC) $(NVCC_FLAGS) -c $^ -o $@
create: matvec_create.c
		
		$(CC) -o $@ $^ $(CFLAGS) -lm
		./create
clean:
		rm *.vec
		rm *.mtx
		rm *.o 
		rm create
		rm minres



