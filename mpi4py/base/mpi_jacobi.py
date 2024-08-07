import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import time

t0 = time.time()
alpha = 0.5
M = 32
N = 16
bound = np.array([-1,1,-1,1]).reshape(2,2)
max_iter = 2000
dx = (bound[0,1] - bound[0,0])/(N - 1)
dy = (bound[1,1] - bound[1,0])/(M - 1)
r1 = -0.5;r2 = -pow(dx/dy,2);r3 = -pow(dy/dx,2);r = 2*(1 - r2 - r3) + alpha*(dx*dx + dy*dy)

def u_acc(x,y):
    return (1.0 - pow(x,2))*(1.0 - pow(y,2))
def f(x,y):
    return 2*(1.0 - pow(x,2)) + 2*(1.0 - pow(y,2)) + alpha*u_acc(x,y)
u = np.zeros(M*N)
u_new = np.zeros_like(u)
u_old = np.zeros_like(u)
f_1d = np.zeros_like(u)

for i in range(M):
    for j in range(N):
        x = bound[0,0] + i*dx
        y = bound[1,0] + j*dy
        u[i*N + j] = u_acc(x,y)

        if i == 0 or i == M - 1 or j == 0 or j == N - 1:
            u_new[i*N + j] = u_acc(x,y)
        else:
            f_1d[i*N + j] = f(x,y)*(dx*dx + dy*dy)
#print(u_new - u)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
step_y = (int)(M/size)
eps = 1e-16
k = 0
while k < max_iter:
    error = 0
    L1_err = 0

    start = rank*step_y
    if (rank + 1)*step_y < M:
        end = (rank + 1)*step_y
    else:
        end = M
    for i in range(start,end):
        for j in range(N):
            u_old[i*N + j] = u_new[i*N + j]
    if size > 1:
        if rank == 0:
            fa_down = (rank + 1)*step_y - 1
            shou_up = (rank + 1)*step_y
            comm.send(u_old[fa_down*N:(fa_down + 1)*N],dest = rank + 1,tag = 0)
            u_old[shou_up*N:(shou_up + 1)*N] = comm.recv(source = rank + 1,tag = 0)
        elif rank > 0 and rank < size - 1:
            fa_up = rank*step_y
            shou_down = rank*step_y - 1
            u_old[shou_down*N:(shou_down + 1)*N] = comm.recv(source = rank - 1, tag = 0)
            comm.send(u_old[fa_up*N:(fa_up + 1)*N],dest = rank - 1, tag = 0)
        
            fa_down = (rank + 1)*step_y - 1
            shou_up = (rank + 1)*step_y
            comm.send(u_old[fa_down*N:(fa_down + 1)*N],dest = rank + 1,tag = 0)
            u_old[shou_up*N:(shou_up + 1)*N] = comm.recv(source = rank + 1,tag = 0)
        else:
            fa_up = rank*step_y
            shou_down = rank*step_y - 1
            u_old[shou_down*N:(shou_down + 1)*N] = comm.recv(source = rank - 1, tag = 0)
            comm.send(u_old[fa_up*N:(fa_up + 1)*N],dest = rank - 1, tag = 0)

    for i in range(start,end):
        for j in range(N):
            if i == 0 or i == M - 1 or j == 0 or j == N - 1:
                continue
            else:
                resid = f_1d[i*N + j] - (r1*u_old[(i - 1)*N + j - 1] + 
                r3*u_old[(i - 1)*N + j] + r1*u_old[(i - 1)*N + j + 1] + 
                r2*u_old[i*N + j - 1] + r*u_old[i*N + j] + 
                r2*u_old[i*N + j + 1] + r1*u_old[(i + 1)*N + j - 1] + 
                r3*u_old[(i + 1)*N + j] + r1*u_old[(i + 1)*N + j + 1])
                u_new[i*N + j] = u_old[i*N + j] + resid/r
                error += resid*resid
            L1_err = max(L1_err,abs(u_new[i*N + j] - u[i*N + j]))
    #print(L1_err)
    if size > 1:
        error = comm.allreduce(error,MPI.SUM)
        L1_err = comm.allreduce(L1_err,MPI.MAX)
    err = np.sqrt(error)
    if err < eps:
        break
    if k%1000 == 0:
        if (rank == 0):
            print("epoch:%d,resid:%.3e,L1_err:%.3e"%(k,err,L1_err))
    k += 1
ela = time.time() - t0
if rank == 0:
    print("Finish at epoch:%d,use rank number = %d,use time:%.2f,L1_err:%.3e"%(k,size,ela,L1_err))
    #print(u_new - u)


