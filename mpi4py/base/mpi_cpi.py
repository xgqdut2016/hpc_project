from mpi4py import MPI
import time
import numpy as np

Kpeat = 1
PI = 3.141592653589793238462643
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if (rank == 0):
    n = 1000000
else:
    n = 0
n = comm.bcast(n,root = 0)

#print("rank = %d,n = %d\n"%(rank,n))
h = 1/n
stragy = 'roundrobin'
#stragy = 'chunk'
chunk = (int)(n/size)
du = n%size
t0 = time.time()
if stragy == 'roundrobin':
    for k in range(Kpeat):
        mysum = 0
        for i in range(rank,n,size):
            x = h*(i + 0.5)
            mysum += np.sqrt(1 - x*x)
        mypi = 4*h*mysum
        pi = comm.allreduce(mypi,MPI.SUM)
else:
    for k in range(Kpeat):
        mysum = 0
        if rank < du:
            for i in range(rank*(chunk + 1),(rank + 1)*(chunk + 1)):
                x = h*(i + 0.5)
                mysum += np.sqrt(1 - x*x)
        else:
            for i in range(du*(chunk + 1) + (rank - du)*chunk,du*(chunk + 1) + (rank + 1 - du)*chunk):
                x = h*(i + 0.5)
                mysum += np.sqrt(1 - x*x)
        mypi = 4*h*mysum
        pi = comm.allreduce(mypi,MPI.SUM)
error = abs(pi - PI)
t1 = time.time()
ela = t1 - t0
if rank == 0:
    print('compute pi:%.10f\n'%(pi))
    print("use rank number:%d,error:%.3e,wall time:%.2f\n"%(size,error,ela))


