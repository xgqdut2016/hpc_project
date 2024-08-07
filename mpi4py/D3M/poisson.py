import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import bfgs
from mpi4py import MPI
import argparse
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
def UU(X, order,prob):
    if prob==1:
        temp = 10*(X[:,0]+X[:,1])**2 + (X[:,0]-X[:,1])**2 + 0.5
        if order[0]==0 and order[1]==0:
            return torch.log(temp)
        if order[0]==1 and order[1]==0:
            return temp**(-1) * (20*(X[:,0]+X[:,1]) + 2*(X[:,0]-X[:,1]))
        if order[0]==0 and order[1]==1:
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
        temp1 = X[:,0]*X[:,0] - X[:,1]*X[:,1]
        temp2 = X[:,0]*X[:,0] + X[:,1]*X[:,1] + 0.1
        if order[0]==0 and order[1]==0:
            return temp1 * temp2**(-1)
        if order[0]==1 and order[1]==0:
            return (2*X[:,0]) * temp2**(-1) + \
                   temp1 * (-1)*temp2**(-2) * (2*X[:,0])
        if order[0]==0 and order[1]==1:
            return (-2*X[:,1]) * temp2**(-1) + \
                   temp1 * (-1)*temp2**(-2) * (2*X[:,1])
        if order[0]==2 and order[1]==0:
            return (2) * temp2**(-1) + \
                   2 * (2*X[:,0]) * (-1)*temp2**(-2) * (2*X[:,0]) + \
                   temp1 * (2)*temp2**(-3) * (2*X[:,0])**2 + \
                   temp1 * (-1)*temp2**(-2) * (2)
        if order[0]==1 and order[1]==1:
            return (2*X[:,0]) * (-1)*temp2**(-2) * (2*X[:,1]) + \
                   (-2*X[:,1]) * (-1)*temp2**(-2) * (2*X[:,0]) + \
                   temp1 * (2)*temp2**(-3) * (2*X[:,0]) * (2*X[:,1])
        if order[0]==0 and order[1]==2:
            return (-2) * temp2**(-1) + \
                   2 * (-2*X[:,1]) * (-1)*temp2**(-2) * (2*X[:,1]) + \
                   temp1 * (2)*temp2**(-3) * (2*X[:,1])**2 + \
                   temp1 * (-1)*temp2**(-2) * (2)
    if prob==3:
        temp = torch.exp(-4*X[:,1]*X[:,1])
        if order[0]==0 and order[1]==0:
            ind = (X[:,0]<=0).float()
            return ind * ((X[:,0]+1)**4-1) * temp + \
                   (1-ind) * (-(-X[:,0]+1)**4+1) * temp
        if order[0]==1 and order[1]==0:
            ind = (X[:,0]<=0).float()
            return ind * (4*(X[:,0]+1)**3) * temp + \
                   (1-ind) * (4*(-X[:,0]+1)**3) * temp
        if order[0]==0 and order[1]==1:
            ind = (X[:,0]<=0).float()
            return ind * ((X[:,0]+1)**4-1) * (temp*(-8*X[:,1])) + \
                   (1-ind) * (-(-X[:,0]+1)**4+1) * (temp*(-8*X[:,1]))
        if order[0]==2 and order[1]==0:
            ind = (X[:,0]<=0).float()
            return ind * (12*(X[:,0]+1)**2) * temp + \
                   (1-ind) * (-12*(-X[:,0]+1)**2) * temp
        if order[0]==1 and order[1]==1:
            ind = (X[:,0]<=0).float()
            return ind * (4*(X[:,0]+1)**3) * (temp*(-8*X[:,1])) + \
                   (1-ind) * (4*(-X[:,0]+1)**3) * (temp*(-8*X[:,1]))
        if order[0]==0 and order[1]==2:
            ind = (X[:,0]<=0).float()
            return ind * ((X[:,0]+1)**4-1) * (temp*(64*X[:,1]*X[:,1]-8)) + \
                   (1-ind) * (-(-X[:,0]+1)**4+1) * (temp*(64*X[:,1]*X[:,1]-8))

def FF(X,prob):
    return -UU(X,[2,0],prob) - UU(X,[0,2],prob)

class INSET():
    def __init__(self,prob,bound,size_tr,size,rank,dtype,dev,para):
        self.dim = bound.shape[0]
        tmp = self.quasi_samples(size_tr)
        tmp[:,0] = tmp[:,0]*(bound[0,1] - bound[0,0]) + bound[0,0]
        tmp[:,1] = tmp[:,1]*(bound[1,1] - bound[1,0]) + bound[1,0]
        self.lap_num = 4
        if size == 1:
            self.X = torch.tensor(tmp).to(dtype)
        else:
            if para == 1:
                self.hl = (bound[0,1] - bound[0,0])/size
                self.lap = self.hl/self.lap_num
                if rank == 0:
                    sub_domain = [bound[0,0] + rank*self.hl,bound[0,0] + (rank + 1)*self.hl + self.lap]
                elif rank == size - 1:
                    sub_domain = [bound[0,0] + rank*self.hl - self.lap,bound[0,0] + (rank + 1)*self.hl]
                else:
                    sub_domain = [bound[0,0] + rank*self.hl - self.lap,bound[0,0] + (rank + 1)*self.hl + self.lap]
                ind = (tmp[:,0] >= sub_domain[0])*(tmp[:,0] <= sub_domain[1])
            elif para == 2:
                self.hl = (bound[1,1] - bound[1,0])/size
                self.lap = self.hl/self.lap_num
                if rank == 0:
                    sub_domain = [bound[1,0] + rank*self.hl,bound[1,0] + (rank + 1)*self.hl + self.lap]
                elif rank == size - 1:
                    sub_domain = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl]
                else:
                    sub_domain = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl + self.lap]
                ind = (tmp[:,1] >= sub_domain[0])*(tmp[:,1] <= sub_domain[1])
            else:
                size_x = 2
                size_y = (int)(size/size_x)
                rank_x = rank%size_x
                rank_y = (int)(rank/size_x)
                self.hl = [(bound[0,1] - bound[0,0])/size_x,(bound[1,1] - bound[1,0])/size_y]
                self.lap = [self.hl[0]/self.lap_num,self.hl[1]/self.lap_num]
                if rank_x == 0:
                    sub_domain_x = [bound[0,0] + rank_x*self.hl[0],bound[0,0] + (rank_x + 1)*self.hl[0] + self.lap[0]]
                    if rank_y == 0:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                    elif rank_y == size_y - 1:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1]]
                    else:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                elif rank_x == size_x - 1:
                    sub_domain_x = [bound[0,0] + rank_x*self.hl[0] - self.lap[0],bound[0,0] + (rank_x + 1)*self.hl[0]]
                    if rank_y == 0:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                    elif rank_y == size_y - 1:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1]]
                    else:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                else:
                    sub_domain_x = [bound[0,0] + rank_x*self.hl[0] - self.lap[0],bound[0,0] + (rank_x + 1)*self.hl[0] + self.lap[0]]
                    if rank_y == 0:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                    elif rank_y == size_y - 1:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1]]
                    else:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                ind_x = (tmp[:,0] >= sub_domain_x[0])*(tmp[:,0] <= sub_domain_x[1])
                ind_y = (tmp[:,1] >= sub_domain_y[0])*(tmp[:,1] <= sub_domain_y[1])
                ind = ind_x*ind_y
            self.X = torch.tensor(tmp[ind]).to(dtype)
        
        self.X = self.X.type(dtype)
        self.X = self.X.to(dev).data
        self.ff = FF(self.X,prob).reshape(-1,1).data
        self.u_acc = UU(self.X,[0,0],prob).reshape(-1,1).data
        
        self.weight_grad = torch.ones_like(self.ff).to(dev)
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample
        

class BDSET():
    def __init__(self,prob,bound,nx_bd,size,rank,dtype,dev,para):
        self.dim = bound.shape[0]
        self.lap_num = 4
        self.nx_bd = nx_bd
        self.X = torch.zeros(2*(nx_bd[0] + nx_bd[1]),2)
        if size == 1:
            self.hx = [(bound[0,1] - bound[0,0])/nx_bd[0],(bound[1,1] - bound[1,0])/nx_bd[1]]
            m = 0
            for i in range(nx_bd[0]):
                self.X[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
                self.X[m,1] = bound[1,0]
                m += 1
            for j in range(nx_bd[1]):
                self.X[m,0] = bound[0,1]
                self.X[m,1] = bound[1,0] + (j + 0.5)*self.hx[1]
                m += 1
            for i in range(nx_bd[0]):
                self.X[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
                self.X[m,1] = bound[1,1]
                m += 1 
            for j in range(nx_bd[1]):
                self.X[m,0] = bound[0,0]
                self.X[m,1] = bound[1,0] + (j + 0.5)*self.hx[1]
                m += 1 
        else:
            if para == 1:
                self.hl = (bound[0,1] - bound[0,0])/size
                self.lap = self.hl/self.lap_num
                if rank == 0:
                    sub_domain_x = [bound[0,0] + rank*self.hl,bound[0,0] + (rank + 1)*self.hl + self.lap]
                elif rank == size - 1:
                    sub_domain_x = [bound[0,0] + rank*self.hl - self.lap,bound[0,0] + (rank + 1)*self.hl]
                else:
                    sub_domain_x = [bound[0,0] + rank*self.hl - self.lap,bound[0,0] + (rank + 1)*self.hl + self.lap]
                sub_domain_y = [bound[1,0],bound[1,1]]
                
            elif para == 2:
                self.hl = (bound[1,1] - bound[1,0])/size
                self.lap = self.hl/self.lap_num
                if rank == 0:
                    sub_domain_y = [bound[1,0] + rank*self.hl,bound[1,0] + (rank + 1)*self.hl + self.lap]
                elif rank == size - 1:
                    sub_domain_y = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl]
                else:
                    sub_domain_y = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl + self.lap]
                sub_domain_x = [bound[0,0],bound[0,1]]
            else:
                size_x = 2
                size_y = (int)(size/size_x)
                rank_x = rank%size_x
                rank_y = (int)(rank/size_x)
                self.hl = [(bound[0,1] - bound[0,0])/size_x,(bound[1,1] - bound[1,0])/size_y]
                self.lap = [self.hl[0]/self.lap_num,self.hl[1]/self.lap_num]
                if rank_x == 0:
                    sub_domain_x = [bound[0,0] + rank_x*self.hl[0],bound[0,0] + (rank_x + 1)*self.hl[0] + self.lap[0]]
                    if rank_y == 0:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                    elif rank_y == size_y - 1:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1]]
                    else:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                elif rank_x == size_x - 1:
                    sub_domain_x = [bound[0,0] + rank_x*self.hl[0] - self.lap[0],bound[0,0] + (rank_x + 1)*self.hl[0]]
                    if rank_y == 0:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                    elif rank_y == size_y - 1:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1]]
                    else:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                else:
                    sub_domain_x = [bound[0,0] + rank_x*self.hl[0] - self.lap[0],bound[0,0] + (rank_x + 1)*self.hl[0] + self.lap[0]]
                    if rank_y == 0:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                    elif rank_y == size_y - 1:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1]]
                    else:
                        sub_domain_y = [bound[1,0] + rank_y*self.hl[1] - self.lap[1],bound[1,0] + (rank_y + 1)*self.hl[1] + self.lap[1]]
                
            self.hx = [(sub_domain_x[1] - sub_domain_x[0])/nx_bd[0],(sub_domain_y[1] - sub_domain_y[0])/nx_bd[1]]
            m = 0
            for i in range(nx_bd[0]):
                self.X[m,0] = sub_domain_x[0] + (i + 0.5)*self.hx[0]
                self.X[m,1] = sub_domain_y[0]
                m += 1
            for j in range(nx_bd[1]):
                self.X[m,0] = sub_domain_x[1]
                self.X[m,1] = sub_domain_y[0] + (j + 0.5)*self.hx[1]
                m += 1
            for i in range(nx_bd[0]):
                self.X[m,0] = sub_domain_x[0] + (i + 0.5)*self.hx[0]
                self.X[m,1] = sub_domain_y[1]
                m += 1 
            for j in range(nx_bd[1]):
                self.X[m,0] = sub_domain_x[0]
                self.X[m,1] = sub_domain_y[0] + (j + 0.5)*self.hx[1]
                m += 1 
        eps = 1e-7
        ind_x = (self.X[:,0] >= bound[0,0] + eps)*(self.X[:,0] <= bound[0,1] - eps)
        ind_y = (self.X[:,1] >= bound[1,0] + eps)*(self.X[:,1] <= bound[1,1] - eps)
        ind = ~(ind_x*ind_y)
        self.X = self.X.type(dtype)
        self.X = self.X.to(dev)
        #plt.scatter(self.X[ind,0].detach().numpy(),self.X[ind,1].detach().numpy());plt.show()
        self.u_acc = torch.zeros_like(self.X[:,0:1])
        self.u_acc[ind] = UU(self.X[ind].detach(),[0,0],prob).reshape(-1,1)
        
np.random.seed(1234)
torch.manual_seed(1234)
class Net(torch.nn.Module):
    def __init__(self, layers, dtype):
        super(Net, self).__init__()
        
        self.layers = layers
        self.layers_hid_num = len(layers)-2
        self.device = device
        self.dtype = dtype
        fc = []
        for i in range(self.layers_hid_num):
            fc.append(torch.nn.Linear(self.layers[i],self.layers[i+1]))
            fc.append(torch.nn.Linear(self.layers[i+1],self.layers[i+1]))
        fc.append(torch.nn.Linear(self.layers[-2],self.layers[-1]))
        self.fc = torch.nn.Sequential(*fc)
        for i in range(self.layers_hid_num):
            self.fc[2*i].weight.data = self.fc[2*i].weight.data.type(dtype)
            self.fc[2*i].bias.data = self.fc[2*i].bias.data.type(dtype)
            self.fc[2*i + 1].weight.data = self.fc[2*i + 1].weight.data.type(dtype)
            self.fc[2*i + 1].bias.data = self.fc[2*i + 1].bias.data.type(dtype)
        self.fc[-1].weight.data = self.fc[-1].weight.data.type(dtype)
        self.fc[-1].bias.data = self.fc[-1].bias.data.type(dtype)
    def forward(self, x):
        dev = x.device
        
        for i in range(self.layers_hid_num):
            h = torch.sin(self.fc[2*i](x))
            h = torch.sin(self.fc[2*i+1](h))
            temp = torch.eye(x.shape[-1],self.layers[i+1],dtype = self.dtype,device = dev)
            x = h + x@temp
        return self.fc[-1](x) 
    def total_para(self):#计算参数数目
        return sum([x.numel() for x in self.parameters()])  

def pred_u(netu,X):
    return netu.forward(X)

def L1_error(u_pred, u_acc):
    u_pred = u_pred.to(device)
    u_acc = u_acc.to(device)
    return max(abs(u_pred - u_acc))


def Loss_fun(netu,inset,bdset):
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.u = pred_u(netu,inset.X)
    bdset.u = pred_u(netu,bdset.X)

    ux, = torch.autograd.grad(inset.u, inset.X, create_graph=True, retain_graph=True,
                                 grad_outputs=inset.weight_grad)
    u_xx, = torch.autograd.grad(ux[:,0:1], inset.X, create_graph=True, retain_graph=True,
                                 grad_outputs=inset.weight_grad)
    u_yy, = torch.autograd.grad(ux[:,1:2], inset.X, create_graph=True, retain_graph=True,
                                 grad_outputs=inset.weight_grad)
    inset.u_lap = u_xx[:,0:1] + u_yy[:,1:2]
    inset.res_u = (inset.u_lap + inset.ff)**2
    inset.pde_u = torch.sqrt(inset.res_u).mean()
    bdset.res_u = (bdset.u - bdset.u_acc.data)**2
    bdset.pde_u = torch.sqrt(bdset.res_u).mean()
    return inset.pde_u + bdset.pde_u
    #return inset.pde_u
def Train(netu,inset,bdset,optimtype,optim,epoch,rank,size):
    if rank == 0:
        print('train neural network')
    t0 = time.time()
    loss = Loss_fun(netu,inset,bdset)
    for it in range(epoch):
        st = time.time()
        if optimtype == 'LBFGS' or optimtype == 'BFGS':
            def closure():
                optim.zero_grad()
                loss = Loss_fun(netu,inset,bdset)
                loss.backward()
                return loss
            optim.step(closure) 
            loss = Loss_fun(netu,inset,bdset)
        else:
            for j in range(100):
                optim.zero_grad()
                loss = Loss_fun(netu,inset,bdset)
                loss.backward()
                optim.step()
        ela = time.time() - st
        if rank == 0:
            print('epoch:%d,loss:%.2e,loss_in:%.2e,loss_bd:%.2e,time:%.2f'%
            (it,loss.item(),inset.pde_u.item(),bdset.pde_u.item(),ela))
    t1 = time.time() - t0
    u_e = L1_error(pred_u(netu,inset.X),inset.u_acc)
    if size > 1:
        if rank > 0:
            comm.send(u_e,dest = 0, tag = rank)
        else:
            for rank in range(1,size):
                ma = comm.recv(source = rank, tag = rank)
                u_e = max(u_e, ma)
            print('use time:%.2f,L1_error of u:%.2e'%(t1,u_e))
    else:
        print('use time:%.2f,L1_error of u:%.2e'%(t1,u_e))
def fun_comm(netu,bdset,rank,size,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2):
    if rank == 0:
        comm.send(bdset.X[ind_st1:ind_ed1,:], dest = big, tag = 0)
        x_right = comm.recv(source = big, tag = 0)
        #-----------------------------
        u_right = pred_u(netu,x_right).detach()
        comm.send(u_right, dest = big, tag = 1)
        bdset.u_acc[ind_st1:ind_ed1,:] = comm.recv(source = big, tag = 1)
    elif rank > 0 and rank < size - 1:
        x_left = comm.recv(source = small, tag = 0)
        comm.send(bdset.X[ind_st2:ind_ed2,:], dest = small, tag = 0)
        #-----------------------------
        u_left = pred_u(netu,x_left).detach()
        bdset.u_acc[ind_st2:ind_ed2,:] = comm.recv(source = small, tag = 1)
        comm.send(u_left, dest = small, tag = 1)
             
        #qian,hou
        comm.send(bdset.X[ind_st1:ind_ed1,:], dest = big, tag = 0)
        x_right = comm.recv(source = big, tag = 0)
        #-----------------------------
        u_right = pred_u(netu,x_right).detach()
        comm.send(u_right, dest = big, tag = 1)
        bdset.u_acc[ind_st1:ind_ed1,:] = comm.recv(source = big, tag = 1)
    else:
        x_left = comm.recv(source = small, tag = 0)
        comm.send(bdset.X[ind_st2:ind_ed2,:], dest = small, tag = 0)
        #-----------------------------
        u_left = pred_u(netu,x_left).detach()
        bdset.u_acc[ind_st2:ind_ed2,:] = comm.recv(source = small, tag = 1)
        comm.send(u_left, dest = small, tag = 1)
def communicate(netu,bdset,rank,size,para):#默认size > 1
    if para == 1:
        big = rank + 1
        small = rank - 1
        ind_st1 = bdset.nx_bd[0]
        ind_ed1 = bdset.nx_bd[0] + bdset.nx_bd[1]
        ind_st2 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
        ind_ed2 = 2*(bdset.nx_bd[0] + bdset.nx_bd[1])
        fun_comm(netu,bdset,rank,size,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)
        
    elif para == 2:
        big = rank + 1
        small = rank - 1
        ind_st1 = bdset.nx_bd[0] + bdset.nx_bd[1]
        ind_ed1 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
        ind_st2 = 0
        ind_ed2 = bdset.nx_bd[0]
        fun_comm(netu,bdset,rank,size,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)
        
    else:
        size_x = 2
        size_y = (int)(size/size_x)
        rank_x = rank%size_x
        rank_y = (int)(rank/size_x)
        if rank_x == 0:
            ind_st = bdset.nx_bd[0]
            ind_ed = bdset.nx_bd[0] + bdset.nx_bd[1]
            comm.send(bdset.X[ind_st:ind_ed,:], dest = rank + 1, tag = 0)
            x_right = comm.recv(source = rank + 1, tag = 0)
            #-----------------------------
            u_right = pred_u(netu,x_right).detach()
            comm.send(u_right, dest = rank + 1, tag = 1)
            bdset.u_acc[ind_st:ind_ed,:] = comm.recv(source = rank + 1, tag = 1)
            #------
            big = rank + size_x
            small = rank - size_x
            ind_st1 = bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed1 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_st2 = 0
            ind_ed2 = bdset.nx_bd[0]
            fun_comm(netu,bdset,rank_y,size_y,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)
            
        elif rank_x > 0 and rank_x < size_x - 1:
            ind_st = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed = 2*(bdset.nx_bd[0] + bdset.nx_bd[1])
            x_left = comm.recv(source = rank - 1, tag = 0)
            comm.send(bdset.X[ind_st:ind_ed,:], dest = rank - 1, tag = 0)
            #-----------------------------
            u_left = pred_u(netu,x_left).detach()
            bdset.u_acc[ind_st:ind_ed,:] = comm.recv(source = rank - 1, tag = 1)
            comm.send(u_left, dest = rank - 1, tag = 1)
             
            ind_st = bdset.nx_bd[0]
            ind_ed = bdset.nx_bd[0] + bdset.nx_bd[1]
            comm.send(bdset.X[ind_st:ind_ed,:], dest = rank + 1, tag = 0)
            x_right = comm.recv(source = rank + 1, tag = 0)
            #-----------------------------
            u_right = pred_u(netu,x_right).detach()
            comm.send(u_right, dest = rank + 1, tag = 1)
            bdset.u_acc[ind_st:ind_ed,:] = comm.recv(source = rank + 1, tag = 1)
            #------
            big = rank + size_x
            small = rank - size_x
            ind_st1 = bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed1 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_st2 = 0
            ind_ed2 = bdset.nx_bd[0]
            fun_comm(netu,bdset,rank_y,size_y,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)
        else:
            ind_st = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed = 2*(bdset.nx_bd[0] + bdset.nx_bd[1])
            x_left = comm.recv(source = rank - 1, tag = 0)
            comm.send(bdset.X[ind_st:ind_ed,:], dest = rank - 1, tag = 0)
            #-----------------------------
            u_left = pred_u(netu,x_left).detach()
            bdset.u_acc[ind_st:ind_ed,:] = comm.recv(source = rank - 1, tag = 1)
            comm.send(u_left, dest = rank - 1, tag = 1)
            #------
            big = rank + size_x
            small = rank - size_x
            ind_st1 = bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed1 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_st2 = 0
            ind_ed2 = bdset.nx_bd[0]
            fun_comm(netu,bdset,rank_y,size_y,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)

bound = np.array([0,1,0,1]).reshape(2,2)               
parser = argparse.ArgumentParser(description='mpi Neural Network Method')
parser.add_argument('--tr', type=int, default=800,
                    help='train size')  
parser.add_argument('--bd', type=int, default=[16,16],
                    help='test size')   
parser.add_argument('--wid', type=int, default=10,
                    help='layers width') 
parser.add_argument('--iter', type=int, default=10,
                    help='max_iter') 
parser.add_argument('--epoch', type=int, default=5,
                    help='initial epoch')                                        
parser.add_argument('--lr', type=float, default=1e0,
                    help='learning rate') 
parser.add_argument('--para', type=int, default=1,
                    help='stragy')
parser.add_argument('--prob', type=int, default=3,
                    help='problem')                                          
dtype = torch.float64
args = parser.parse_args()
size_tr = args.tr
nx_bd = args.bd
lay_wid = args.wid
max_iters = args.iter
epoch = args.epoch  
lr = args.lr
para = args.para
prob = args.prob

layer_u = [2,lay_wid,lay_wid,1]                  
netu = Net(layer_u,dtype).to(device)
fname = 'size-rank-(%d,%d)unet.pt'%(size,rank)
#optimtype = 'BFGS'
optimtype = 'LBFGS'
#optimtype = 'adam'
if optimtype == 'BFGS':
    optim = bfgs.BFGS(netu.parameters(),lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
elif optimtype == 'LBFGS':
    optim = torch.optim.LBFGS(netu.parameters(),lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
else:
    optim = torch.optim.Adam(netu.parameters(),lr=lr)          
inset = INSET(prob,bound,size_tr,size,rank,dtype,device,para)   
bdset = BDSET(prob,bound,nx_bd,size,rank,dtype,device,para)   
#plt.scatter(inset.X[:,0].detach().numpy(),inset.X[:,1].detach().numpy()) ;plt.show()
#print(Loss_fun(netu,inset,bdset))
start_time = time.time()
for i in range(max_iters):
    
    if rank == 0:
        print('\n    Iters: %d' %(i))
    Train(netu,inset,bdset,optimtype,optim,epoch,rank,size)
    if size > 1:
        communicate(netu,bdset,rank,size,para)
    

ela = time.time() - start_time
nx_te = [21,21]
x_train = np.linspace(bound[0,0],bound[0,1],nx_te[0])
y_train = np.linspace(bound[1,0],bound[1,1],nx_te[1])
inp = torch.zeros(nx_te[0]*nx_te[1],2).type(dtype)
for i in range(nx_te[0]):
    for j in range(nx_te[1]):
        inp[j*nx_te[0] + i,0] = x_train[i]
        inp[j*nx_te[0] + i,1] = y_train[j]

def choose(netu,para,size,rank,inp):
    u_pred = np.zeros_like(inp[:,0:1])
    if para == 1:
        hl = (bound[0,1] - bound[0,0])/size
        sub_domain = [bound[0,0] + rank*hl,bound[0,0] + (rank + 1)*hl]
        ind = (inp[:,0] >= sub_domain[0])*(inp[:,0] <= sub_domain[1])
        
    elif para == 2:
        hl = (bound[1,1] - bound[1,0])/size
        sub_domain = [bound[1,0] + rank*hl,bound[1,0] + (rank + 1)*hl]
        ind = (inp[:,1] >= sub_domain[0])*(inp[:,1] <= sub_domain[1])
        
    else:
        size_x = 2
        size_y = (int)(size/size_x)
        rank_x = rank%size_x
        rank_y = (int)(rank/size_x)
        hl = [(bound[0,1] - bound[0,0])/size_x,(bound[1,1] - bound[1,0])/size_y]
        sub_domain_x = [bound[0,0] + rank_x*hl[0],bound[0,0] + (rank_x + 1)*hl[0]]
        sub_domain_y = [bound[1,0] + rank_y*hl[1],bound[1,0] + (rank_y + 1)*hl[1]]
        ind_x = (inp[:,0] >= sub_domain_x[0])*(inp[:,0] <= sub_domain_x[1])
        ind_y = (inp[:,1] >= sub_domain_y[0])*(inp[:,1] <= sub_domain_y[1])
        ind = ind_x*ind_y
    u_pred[ind] = pred_u(netu,inp[ind]).detach().numpy()
    return ind,u_pred
ind,u_pred = choose(netu,para,size,rank,inp)
if rank > 0:
    comm.send(ind,dest = 0, tag = 0)
    comm.send(u_pred, dest = 0, tag = 1)
else:
    for rank in range(1,size):
        ind = comm.recv(source = rank,tag = 0)
        tmp = comm.recv(source = rank, tag = 1)
        u_pred[ind] = tmp[ind]
    num_line = 40
    cma = 'rainbow'
    x,y = np.meshgrid(x_train,y_train)
    err = u_pred - UU(inp,[0,0],prob).reshape(-1,1).detach().numpy()
    
    fig, ax = plt.subplots(figsize=(27,18))
    s = ax.contourf(x,y,err.reshape(nx_te[0],nx_te[1]), num_line, cmap=cma)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    fig.colorbar(s,ax=ax,format = '%.2e')
    ax.set_xlabel('x',fontsize = 25)
    ax.set_ylabel('y',fontsize = 25)
    ax.set_title('prob:%d'%(prob))
    plt.savefig('parallel.png')
    plt.show()


