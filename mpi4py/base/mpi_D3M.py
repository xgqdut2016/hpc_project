import matplotlib.pyplot as plt
import numpy as np
import time
from mpi4py import MPI

M = 16
N = 32
bound = np.array([-1,1,-1,1]).reshape(2,2)
max_iter = 100
dx = (bound[0,1] - bound[0,0])/(M - 1)
dy = (bound[1,1] - bound[1,0])/(N - 1)
prob = 1

def UU(X, order,prob):#X表示(x,t)
    if prob==1:
        temp = 10*(X[:,0]+X[:,1])**2 + (X[:,0]-X[:,1])**2 + 0.5
        if order[0]==0 and order[1]==0:
            return np.log(temp)
        if order[0]==1 and order[1]==0:#对x求偏导
            return temp**(-1) * (20*(X[:,0]+X[:,1]) + 2*(X[:,0]-X[:,1]))
        if order[0]==0 and order[1]==1:#对t求偏导
            return temp**(-1) * (20*(X[:,0]+X[:,1]) - 2*(X[:,0]-X[:,1]))
        if order[0]==2 and order[1]==0:
            return - temp**(-2) * (20*(X[:,0]+X[:,1])+2*(X[:,0]-X[:,1])) ** 2 \
                   + temp**(-1) * (22)
        if order[0]==1 and order[1]==1:
            return - temp**(-2) * (20*(X[:,0]+X[:,1])+2*(X[:,0]-X[:,1])) \
                   * (20*(X[:,0]+X[:,1])-2*(X[:,0]-X[:,1])) \
                   + temp**(-1) * (18)
        if order[0]==0 and order[1]==2:
            return - temp**(-2) * (20*(X[:,0]+X[:,1])-2*(X[:,0]-X[:,1])) ** 2 \
                   + temp**(-1) * (22)
    if prob==2:
        if order[0]==0 and order[1]==0:
            return (X[:,0]*X[:,0]*X[:,0]-X[:,0]) * \
                   0.5*(np.exp(2*X[:,1])+np.exp(-2*X[:,1]))
        if order[0]==1 and order[1]==0:
            return (3*X[:,0]*X[:,0]-1) * \
                   0.5*(np.exp(2*X[:,1])+np.exp(-2*X[:,1]))
        if order[0]==0 and order[1]==1:
            return (X[:,0]*X[:,0]*X[:,0]-X[:,0]) * \
                   (np.exp(2*X[:,1])-np.exp(-2*X[:,1]))
        if order[0]==2 and order[1]==0:
            return (6*X[:,0]) * \
                   0.5*(np.exp(2*X[:,1])+np.exp(-2*X[:,1]))
        if order[0]==1 and order[1]==1:
            return (3*X[:,0]*X[:,0]-1) * \
                   (np.exp(2*X[:,1])-np.exp(-2*X[:,1]))
        if order[0]==0 and order[1]==2:
            return (X[:,0]*X[:,0]*X[:,0]-X[:,0]) * \
                   2*(np.exp(2*X[:,1])+np.exp(-2*X[:,1]))


def FF(prob,X):
    return -UU(X,[0,2],prob) - UU(X,[2,0],prob)

def omega_mat(left_ind,right_ind,down_ind,up_ind):
    m = right_ind - left_ind + 1
    n = up_ind - down_ind + 1
    A = np.zeros([m*n,m*n])
    for j in range(n):
        for i in range(m):
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                A[j*m + i][j*m + i] = 1
            else:
                A[j*m + i][j*m + i] = 2*(dx/dy + dy/dx)
                A[j*m + i,(j - 1)*m + i] = -dx/dy
                A[j*m + i,(j + 1)*m + i] = -dx/dy
                A[j*m + i,j*m + i - 1] = -dy/dx
                A[j*m + i,j*m + i + 1] = -dy/dx
    return A
def oemga_rig(left_ind,right_ind,down_ind,up_ind,prob,u_l,u_r,u_d,u_u):
    m = right_ind - left_ind + 1
    n = up_ind - down_ind + 1
    b = np.zeros([m*n,1])
    for j in range(n):
        for i in range(m):
            x = bound[0,0] + (i + left_ind)*dx
            y = bound[1,0] + (j + down_ind)*dy
            X = np.array([[x,y]])
            if i == 0:
                b[j*m + i][0] = u_l[j]
            elif i == m - 1:
                b[j*m + i][0] = u_r[j]
            elif j == 0:
                b[j*m + i][0] = u_d[i]
            elif j == n - 1:
                b[j*m + i][0] = u_u[i]
            else:
                b[j*m + i][0] = FF(prob,X)*dx*dy
    return b
def init_data(prob,M,N):
    u_acc = np.zeros([M*N,1])
    dirchlet = np.zeros([M*N,1])
    
    for j in range(N):
        for i in range(M):
            x = bound[0,0] + i*dx
            y = bound[1,0] + j*dy
            X = np.array([[x,y]])
            u_acc[j*M + i,0] = UU(X,[0,0],prob)
    return u_acc


u_acc = init_data(prob,M,N)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
step_y = (int)(N/size)

if size == 1:
    left_ind = 0
    right_ind = M - 1
    down_ind = 0
    up_ind = N - 1
    u_l = np.zeros(N)
    u_r = np.zeros(N)
    u_d = np.zeros(M)
    u_u = np.zeros(M)
    for j in range(N):
        X = np.array([[bound[0,0],bound[1,0] + j*dy]])
        u_l[j] = UU(X,[0,0],prob)
        X = np.array([[bound[0,1],bound[1,0] + j*dy]])
        u_r[j] = UU(X,[0,0],prob)
    for i in range(M):
        X = np.array([[bound[0,0] + i*dx,bound[1,0]]])
        u_d[i] = UU(X,[0,0],prob)
        X = np.array([[bound[0,0] + i*dx,bound[1,1]]])
        u_u[i] = UU(X,[0,0],prob)
elif size == 2:
    if rank == 0:
        left_ind = 0
        right_ind = M - 1
        down_ind = 0
        up_ind = step_y + 1
        u_l = np.zeros(step_y + 2)
        u_r = np.zeros(step_y + 2)
        u_d = np.zeros(M)
        u_u = np.zeros(M)
        for j in range(step_y + 2):
            X = np.array([[bound[0,0],bound[1,0] + j*dy]])
            u_l[j] = UU(X,[0,0],prob)
            X = np.array([[bound[0,1],bound[1,0] + j*dy]])
            u_r[j] = UU(X,[0,0],prob)
        for i in range(M):
            X = np.array([[bound[0,0] + i*dx,bound[1,0]]])
            u_d[i] = UU(X,[0,0],prob)
            X = np.array([[bound[0,0] + i*dx,bound[1,1]]])
            u_u[i] = UU(X,[0,0],prob)

    else:
        left_ind = 0
        right_ind = M - 1
        down_ind = step_y
        up_ind = N - 1
        u_l = np.zeros(N - step_y)
        u_r = np.zeros(N - step_y )
        u_d = np.zeros(M)
        u_u = np.zeros(M)
        for j in range(N - step_y):
            X = np.array([[bound[0,0],bound[1,0] + (j + step_y)*dy]])
            u_l[j] = UU(X,[0,0],prob)
            X = np.array([[bound[0,1],bound[1,0] + (j + step_y)*dy]])
            u_r[j] = UU(X,[0,0],prob)
        for i in range(M):
            X = np.array([[bound[0,0] + i*dx,bound[1,0]]])
            u_d[i] = UU(X,[0,0],prob)
            X = np.array([[bound[0,0] + i*dx,bound[1,1]]])
            u_u[i] = UU(X,[0,0],prob)
else:
    if rank == 0:
        left_ind = 0
        right_ind = M - 1
        down_ind = 0
        up_ind = step_y
        u_l = np.zeros(step_y + 1)
        u_r = np.zeros(step_y + 1)
        u_d = np.zeros(M)
        u_u = np.zeros(M)
        for j in range(step_y + 1):
            X = np.array([[bound[0,0],bound[1,0] + j*dy]])
            u_l[j] = UU(X,[0,0],prob)
            X = np.array([[bound[0,1],bound[1,0] + j*dy]])
            u_r[j] = UU(X,[0,0],prob)
        for i in range(M):
            X = np.array([[bound[0,0] + i*dx,bound[1,0]]])
            u_d[i] = UU(X,[0,0],prob)
            X = np.array([[bound[0,0] + i*dx,bound[1,1]]])
            u_u[i] = UU(X,[0,0],prob)

    elif rank == size - 1:
        left_ind = 0
        right_ind = M - 1
        down_ind = step_y*rank - 1
        up_ind = N - 1
        u_l = np.zeros(N - step_y*rank + 1)
        u_r = np.zeros(N - step_y*rank + 1)
        u_d = np.zeros(M)
        u_u = np.zeros(M)
        for j in range(N - step_y*rank + 1):
            X = np.array([[bound[0,0],bound[1,0] + (j + step_y*rank - 1)*dy]])
            u_l[j] = UU(X,[0,0],prob)
            X = np.array([[bound[0,1],bound[1,0] + (j + step_y*rank - 1)*dy]])
            u_r[j] = UU(X,[0,0],prob)
        for i in range(M):
            X = np.array([[bound[0,0] + i*dx,bound[1,0]]])
            u_d[i] = UU(X,[0,0],prob)
            X = np.array([[bound[0,0] + i*dx,bound[1,1]]])
            u_u[i] = UU(X,[0,0],prob)
    else:
        left_ind = 0
        right_ind = M - 1
        down_ind = step_y*rank - 1
        up_ind = step_y*(rank + 1)
        u_l = np.zeros(step_y + 2)
        u_r = np.zeros(step_y + 2)
        u_d = np.zeros(M)
        u_u = np.zeros(M)
        for j in range(step_y + 2):
            X = np.array([[bound[0,0],bound[1,0] + (j + step_y*rank - 1)*dy]])
            u_l[j] = UU(X,[0,0],prob)
            X = np.array([[bound[0,1],bound[1,0] + (j + step_y*rank - 1)*dy]])
            u_r[j] = UU(X,[0,0],prob)
        for i in range(M):
            X = np.array([[bound[0,0] + i*dx,bound[1,0]]])
            u_d[i] = UU(X,[0,0],prob)
            X = np.array([[bound[0,0] + i*dx,bound[1,1]]])
            u_u[i] = UU(X,[0,0],prob)



eps = 1e-10
k = 0
t0 = time.time()
while k < max_iter:
    if size == 1:
        
        A = omega_mat(left_ind,right_ind,down_ind,up_ind)
        b = oemga_rig(left_ind,right_ind,down_ind,up_ind,prob,u_l,u_r,u_d,u_u)
        #print(A.shape,b.shape)
        u_pred = np.linalg.solve(A,b)

        L1_err = max(abs(u_pred - u_acc))
        
        break
    elif size == 2:
        if rank == 0:
            A = omega_mat(left_ind,right_ind,down_ind,up_ind)
            b = oemga_rig(left_ind,right_ind,down_ind,up_ind,prob,u_l,u_r,u_d,u_u)
            u_pred = np.linalg.solve(A,b)
            L1_err = max(abs(u_pred - u_acc[down_ind*M:(up_ind + 1)*M]))
            comm.send(u_pred[step_y*M:(step_y + 1)*M],dest = 1,tag = 0)
            u_u = comm.recv(source = 1,tag = 0)
            err_rank = max(abs(u_u - u_pred[(step_y + 1)*M:(step_y + 2)*M]))
            #print(k,u_l.shape,u_r.shape,u_d.shape,u_u.shape)
        else:
            
            A = omega_mat(left_ind,right_ind,down_ind,up_ind)
            b = oemga_rig(left_ind,right_ind,down_ind,up_ind,prob,u_l,u_r,u_d,u_u)
            u_pred = np.linalg.solve(A,b)
            L1_err = max(abs(u_pred - u_acc[down_ind*M:(up_ind + 1)*M]))
            u_d = comm.recv(source = 0,tag = 0)
            #print(u_pred[(step_y + 1)*M:(step_y + 2)*M].shape)
            comm.send(u_pred[M:2*M],dest = 0,tag = 0)
            err_rank = max(abs(u_d - u_pred[:M]))
            #print(u_pred[(step_y + 1)*M:(step_y + 2)*M].shape,M,(step_y + 1)*M,step_y + 1,u_pred.shape)
        L1_err = comm.allreduce(L1_err,MPI.MAX)
        err_rank = comm.allreduce(err_rank,MPI.MAX)
    else:
        if rank == 0:
            A = omega_mat(left_ind,right_ind,down_ind,up_ind)
            b = oemga_rig(left_ind,right_ind,down_ind,up_ind,prob,u_l,u_r,u_d,u_u)
            u_pred = np.linalg.solve(A,b)
            L1_err = max(abs(u_pred - u_acc[down_ind*M:(up_ind + 1)*M]))
            comm.send(u_pred[(step_y - 1)*M:(step_y)*M],dest = 1,tag = 0)
            u_u = comm.recv(source = 1,tag = 0)
            err_rank = max(abs(u_u - u_pred[(step_y)*M:(step_y + 1)*M]))
        elif rank > 0 and rank < size - 1:
            A = omega_mat(left_ind,right_ind,down_ind,up_ind)
            b = oemga_rig(left_ind,right_ind,down_ind,up_ind,prob,u_l,u_r,u_d,u_u)
            u_pred = np.linalg.solve(A,b)
            L1_err = max(abs(u_pred - u_acc[down_ind*M:(up_ind + 1)*M]))
            u_d = comm.recv(source = rank - 1,tag = 0)
            err_rank_d = max(abs(u_d - u_pred[:M]))
            comm.send(u_pred[M:2*M],dest = rank - 1,tag = 0)
            comm.send(u_pred[step_y*M:(step_y + 1)*M],dest = rank + 1,tag = 0)
            u_u = comm.recv(source = rank + 1,tag = 0)
            err_rank_u = max(abs(u_u - u_pred[(step_y + 1)*M:(step_y + 2)*M]))
            err_rank = max(err_rank_d,err_rank_u)
        else:
            A = omega_mat(left_ind,right_ind,down_ind,up_ind)
            b = oemga_rig(left_ind,right_ind,down_ind,up_ind,prob,u_l,u_r,u_d,u_u)
            u_pred = np.linalg.solve(A,b)
            L1_err = max(abs(u_pred - u_acc[down_ind*M:(up_ind + 1)*M]))
            u_d = comm.recv(source = rank - 1,tag = 0)
            err_rank = max(abs(u_d - u_pred[:M]))
            comm.send(u_pred[M:2*M],dest = rank - 1,tag = 0)
        err_rank = comm.allreduce(err_rank,MPI.MAX)
        L1_err = comm.allreduce(L1_err,MPI.MAX)
    if err_rank < eps:
        break
    if k%50 == 0:
        if rank == 0 and size > 1:
            print("epoch:%d,err_rank:%.3e,L1_err:%.3e"%(k,err_rank,L1_err))
    k += 1
ela = time.time() - t0
if rank == 0:
    print("Finish at epoch:%d,use rank number = %d,use time:%.2f,L1_err:%.3e"%(k,size,ela,L1_err))
    #print(u_new - u)

            




