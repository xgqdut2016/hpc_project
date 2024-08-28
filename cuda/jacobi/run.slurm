#!/bin/bash

#SBATCH -o job_%j_%N.out
#SBATCH --partition=gpu
#SBATCH -J vec_add
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH -t 10:00

./jacobi 32
./jacobi 64
./jacobi 128

