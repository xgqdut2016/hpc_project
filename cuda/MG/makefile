CC = gcc
CFLAGS = -std=c99

NVCC = nvcc
NVCC_FLAGS = -std=c++11 -O3 -Wno-deprecated-gpu-targets

LIBRARIES = -L/${CUDA_DIR}/lib64 -lcudart -lm

EXE = mg
all: ${EXE}

mg: main.o cuda_mg.o
		$(CC) $^ -o $@ $(LIBRARIES)

main.o: main.c
		$(CC) $(CFLAGS) -c $^ -o $@

cuda_mg.o: cuda_mg.cu
		$(NVCC) $(NVCC_FLAGS) -c $^ -o $@

clean:
		rm *.o







