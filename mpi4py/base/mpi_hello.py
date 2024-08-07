from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    msg = 'Hello, world'
    comm.send(msg, dest=1)
elif rank == 1:
    s = comm.recv()
    print("rank %d: %s" % (rank, s))
else:
    print("rank %d: idle" % (rank))
M = 3
N = 7
tmp = np.empty(N)
u = np.random.rand(M*N)
if rank == 0:
    
    comm.send(u[:N],dest = rank + 1,tag = 0)
    y = np.array([0,1,2,3,4,5,6])
    comm.send(y,dest = rank + 1,tag = 1)
    print(u)
elif rank == 1:
    tmp = comm.recv(source = rank - 1,tag = 0)
    print(tmp)
    u[1*N:2*N] = comm.recv(source = rank - 1,tag = 1)
    print(u)




