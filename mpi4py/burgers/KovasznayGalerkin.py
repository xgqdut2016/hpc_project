import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import bfgs
from mpi4py import MPI
import argparse
import itertools
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
bound = np.array([-0.5,1,-0.5,1.5]).reshape(2,2)
Re = 40;nu = 1/Re
lam = 1/(2*nu) - np.sqrt(1/(4*nu**2) + 4*np.pi**2)
def UU(X,order,prob):
    if prob == 1:
        eta = 2*np.pi
        if order == [0,0]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = 1 - torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)
            tmp[:,1] = torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*lam/(eta)
            return tmp
        if order == [1,0]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = - torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)*lam
            tmp[:,1] = torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*(lam**2)/(eta)
            return tmp
        if order == [0,1]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*(eta)
            tmp[:,1] = torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)*lam
            return tmp
        if order == [2,0]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = - torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)*lam*lam
            tmp[:,1] = torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*(lam**3)/(eta)
            return tmp
        if order == [0,2]:
            tmp = torch.zeros(X.shape[0],2)
            tmp[:,0] = torch.exp(lam*X[:,0])*torch.cos(X[:,1]*eta)*(eta)**2
            tmp[:,1] = -torch.exp(lam*X[:,0])*torch.sin(X[:,1]*eta)*lam*(eta)
            return tmp
    
def Delta(X,prob):
    return UU(X,[2,0],prob) + UU(X,[0,2],prob)
def PP(X,order,prob):
    if prob == 1:
        if order == [0,0]:
            return 0.5*(1 - torch.exp(2*lam*X[:,0]))
        if order == [1,0]:
            return - lam*torch.exp(2*lam*X[:,0])
        if order == [0,1]:
            return 0*X[:,0]
def FF(X,prob):#mu = 0.5
    tmp = torch.zeros(X.shape[0],2)
    tmp[:,0] = -nu*Delta(X,prob)[:,0] + (UU(X,[0,0],prob)[:,0])*(UU(X,[1,0],prob)[:,0]) + \
    (UU(X,[0,0],prob)[:,1])*(UU(X,[0,1],prob)[:,0]) + PP(X,[1,0],prob)
    tmp[:,1] = -nu*Delta(X,prob)[:,1] + (UU(X,[0,0],prob)[:,0])*(UU(X,[1,0],prob)[:,1]) + \
    (UU(X,[0,0],prob)[:,1])*(UU(X,[0,1],prob)[:,1]) + PP(X,[0,1],prob)
    return tmp

class INSET():
    def __init__(self,prob,bound,size_tr,size,rank,dtype,dev,para):
        self.dim = bound.shape[0]
        self.hr = 1e-3
        tmp = self.quasi_samples(size_tr)
        tmp[:,0] = tmp[:,0]*(bound[0,1] - bound[0,0] - 2*self.hr) + bound[0,0] + self.hr
        tmp[:,1] = tmp[:,1]*(bound[1,1] - bound[1,0] - 2*self.hr) + bound[1,0] + self.hr
        self.lap_num = 4
        if size == 1:
            self.Node = torch.tensor(tmp).to(dtype)
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
                ind = (tmp[:,0] >= sub_domain[0] + self.hr)*(tmp[:,0] <= sub_domain[1] - self.hr)
            elif para == 2:
                self.hl = (bound[1,1] - bound[1,0])/size
                self.lap = self.hl/self.lap_num
                if rank == 0:
                    sub_domain = [bound[1,0] + rank*self.hl,bound[1,0] + (rank + 1)*self.hl + self.lap]
                elif rank == size - 1:
                    sub_domain = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl]
                else:
                    sub_domain = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl + self.lap]
                ind = (tmp[:,1] >= sub_domain[0] + self.hr)*(tmp[:,1] <= sub_domain[1] - self.hr)
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
                ind_x = (tmp[:,0] >= sub_domain_x[0] + self.hr)*(tmp[:,0] <= sub_domain_x[1] - self.hr)
                ind_y = (tmp[:,1] >= sub_domain_y[0] + self.hr)*(tmp[:,1] <= sub_domain_y[1] - self.hr)
                ind = ind_x*ind_y
            self.Node = torch.tensor(tmp[ind]).to(dtype)
        num = self.Node.shape[0]
        self.X = torch.zeros(num,4,2)
        for i in range(num):
            self.X[i,0,0] = self.Node[i,0] + 0.5*self.hr
            self.X[i,0,1] = self.Node[i,1] + 0.5*self.hr
            self.X[i,1,0] = self.Node[i,0] - 0.5*self.hr
            self.X[i,1,1] = self.Node[i,1] + 0.5*self.hr
            self.X[i,2,0] = self.Node[i,0] + 0.5*self.hr
            self.X[i,2,1] = self.Node[i,1] - 0.5*self.hr
            self.X[i,3,0] = self.Node[i,0] - 0.5*self.hr
            self.X[i,3,1] = self.Node[i,1] - 0.5*self.hr

        self.v = torch.zeros(self.Node.shape[0],4,1)
        self.ff = torch.zeros(self.Node.shape[0],4,2)
        self.uu = torch.zeros(self.Node.shape[0],4,1)
        self.vv = torch.zeros(self.Node.shape[0],4,1)
        for i in range(self.Node.shape[0]):
            self.v[i,:,0:1] = self.basic(self.X[i,:,:],[0,0],i).view(-1,1)
            self.ff[i,:,0:1] = FF(self.X[i,:,:],prob)[:,0:1]
            self.ff[i,:,1:2] = FF(self.X[i,:,:],prob)[:,1:2]
            self.uu[i,:,0:1] = UU(self.X[i,:,:],[0,0],prob)[:,0:1]
            self.vv[i,:,0:1] = UU(self.X[i,:,:],[0,0],prob)[:,1:2]
        self.vx = torch.zeros(self.Node.shape[0],4,self.dim)
        for i in range(self.Node.shape[0]):
            self.vx[i,:,0] = self.basic(self.X[i,:,:],[1,0],i)
            self.vx[i,:,1] = self.basic(self.X[i,:,:],[0,1],i)
        self.X = self.X.type(dtype)
        self.v = self.v.type(dtype)
        self.vx = self.vx.type(dtype)
        self.ff = self.ff.type(dtype)
        self.uu = self.uu.type(dtype)
        self.vv = self.vv.type(dtype)
        self.X = self.X.to(dev).data
        self.v = self.v.to(dev).data
        self.vx = self.vx.to(dev).data
        self.ff = self.ff.to(dev).data
        self.uu = self.uu.to(dev).data
        self.vv = self.vv.to(dev).data
        self.weight_grad = torch.ones_like(self.uu).to(dev)
    def phi(self,X,order):#[-1,1]*[-1,1]，在原点取值为1，其他网格点取值为0的基函数
        ind00 = (X[:,0] >= -1);ind01 = (X[:,0] >= 0);ind02 = (X[:,0] >= 1)
        ind10 = (X[:,1] >= -1);ind11 = (X[:,1] >= 0);ind12 = (X[:,1] >= 1)
        if order == [0,0]:
            return (ind00*~ind01*ind10*~ind11).float()*(1 + X[:,0])*(1 + X[:,1]) + \
                    (ind01*~ind02*ind10*~ind11).float()*(1 - X[:,0])*(1 + X[:,1]) + \
                    (ind00*~ind01*ind11*~ind12).float()*(1 + X[:,0])*(1 - X[:,1]) + \
                    (ind01*~ind02*ind11*~ind12).float()*(1 - X[:,0])*(1 - X[:,1])
        if order == [1,0]:
            return (ind00*~ind01*ind10*~ind11).float()*(1 + X[:,1]) + \
                    (ind01*~ind02*ind10*~ind11).float()*(-(1 + X[:,1])) + \
                    (ind00*~ind01*ind11*~ind12).float()*(1 - X[:,1]) + \
                    (ind01*~ind02*ind11*~ind12).float()*(-(1 - X[:,1]))
        if order == [0,1]:
            return (ind00*~ind01*ind10*~ind11).float()*(1 + X[:,0]) + \
                    (ind01*~ind02*ind10*~ind11).float()*(1 - X[:,0]) + \
                    (ind00*~ind01*ind11*~ind12).float()*(-(1 + X[:,0])) + \
                    (ind01*~ind02*ind11*~ind12).float()*(-(1 - X[:,0]))
    def basic(self,X,order,i):#根据网格点的存储顺序，遍历所有网格点，取基函数
        temp = (X - self.Node[i,:])/torch.tensor([self.hr,self.hr])
        if order == [0,0]:
            return self.phi(temp,order)
        if order == [1,0]:
            return self.phi(temp,order)/self.hr
        if order == [0,1]:
            return self.phi(temp,order)/self.hr
        
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
        self.uu = torch.zeros_like(self.X[:,0:1])
        self.vv = torch.zeros_like(self.X[:,0:1])
        self.uu[ind] = UU(self.X[ind].detach(),[0,0],prob)[:,0:1]
        self.vv[ind] = UU(self.X[ind].detach(),[0,0],prob)[:,1:2]
        self.X = self.X.type(dtype)
        self.X = self.X.to(dev).data
        self.uu = self.uu.type(dtype)
        self.uu = self.uu.to(dev).data
        self.vv = self.vv.type(dtype)
        self.vv = self.vv.to(dev).data
        
        #plt.scatter(self.X[ind,0].detach().numpy(),self.X[ind,1].detach().numpy());plt.show()
                
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
def pred_v(netv,X):
    return netv.forward(X)
def pred_p(netp,X):
    return netp.forward(X)

def L1_error(u_pred, u_acc):
    u_pred = u_pred.to(device).reshape(-1,1)
    u_acc = u_acc.to(device).reshape(-1,1)
    return max(abs(u_pred - u_acc))

def Lossyp(netu,netv,netp,inset,bdset):
    if inset.X.requires_grad == False:
        inset.X.requires_grad = True
    inset.u = pred_u(netu,inset.X)
    inset.v = pred_v(netv,inset.X)
    inset.p = pred_p(netp,inset.X)
    u_x, = torch.autograd.grad(inset.u, inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=inset.weight_grad)
    u_xx, = torch.autograd.grad(u_x[:,:,0:1], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=inset.weight_grad)
    u_yy, = torch.autograd.grad(u_x[:,:,1:2], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=inset.weight_grad)  
    u_lap = u_xx[:,:,0:1] + u_yy[:,:,1:2]

    v_x, = torch.autograd.grad(inset.v, inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=inset.weight_grad)
    v_xx, = torch.autograd.grad(v_x[:,:,0:1], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=inset.weight_grad)
    v_yy, = torch.autograd.grad(v_x[:,:,1:2], inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=inset.weight_grad)  
    v_lap = v_xx[:,:,0:1] + v_yy[:,:,1:2]

    p_x, = torch.autograd.grad(inset.p, inset.X, create_graph=True, retain_graph=True,
                               grad_outputs=inset.weight_grad)
    u_in = nu*(u_x*inset.vx).sum(2) - (inset.vx[:,:,0:1]*inset.p).sum(2) + \
        ((inset.u*u_x[:,:,0:1] + inset.v*u_x[:,:,1:2] - inset.ff[:,:,0:1])*inset.v).sum(2)
    v_in = nu*(v_x*inset.vx).sum(2) - (inset.vx[:,:,1:2]*inset.p).sum(2) + \
        ((inset.u*v_x[:,:,0:1] + inset.v*v_x[:,:,1:2] - inset.ff[:,:,1:2])*inset.v).sum(2)
    div_in = (inset.u*inset.vx[:,:,0:1]).sum(2) + (inset.v*inset.vx[:,:,1:2]).sum(2)
    inset.res_u = u_in.mean(1)**2
    inset.res_v = v_in.mean(1)**2
    inset.res_div = div_in.mean(1)**2
    #inset.res_u = (-nu*u_lap + inset.u*u_x[:,:,0:1] + inset.v*u_x[:,:,1:2] + p_x[:,:,0:1] - inset.ff[:,:,0:1])**2
    #inset.res_v = (-nu*v_lap + inset.u*v_x[:,:,0:1] + inset.v*v_x[:,:,1:2] + p_x[:,:,1:2] - inset.ff[:,:,1:2])**2
    #inset.res_div = (u_x[:,:,0:1] + v_x[:,:,1:2])**2
    inset.loss_u = torch.sqrt(inset.res_u.mean())
    inset.loss_v = torch.sqrt(inset.res_v.mean())
    inset.loss_div = torch.sqrt(inset.res_div.mean())

    inset.loss = inset.loss_u + inset.loss_v + inset.loss_div
    bdset.res_u = (pred_u(netu,bdset.X) - bdset.uu)**2
    bdset.res_v = (pred_v(netv,bdset.X) - bdset.vv)**2
    bdset.loss = torch.sqrt(bdset.res_u).mean() + torch.sqrt(bdset.res_v).mean()
    return inset.loss + bdset.loss

def Train(netu,netv,netp,inset,bdset,optimtype,optim,epoch,rank,size):
    if rank == 0:
        print('train neural network')
    t0 = time.time()
    loss = Lossyp(netu,netv,netp,inset,bdset)
    for it in range(epoch):
        st = time.time()
        if optimtype == 'LBFGS' or optimtype == 'BFGS':
            def closure():
                optim.zero_grad()
                loss = Lossyp(netu,netv,netp,inset,bdset)
                loss.backward()
                return loss
            optim.step(closure) 
        else:
            for j in range(100):
                optim.zero_grad()
                loss = Lossyp(netu,netv,netp,inset,bdset)
                loss.backward()
                optim.step()
        loss = Lossyp(netu,netv,netp,inset,bdset)
        ela = time.time() - st
        if rank == 0:
            print('epoch:%d,loss:%.2e,loss_in:%.2e,loss_bd:%.2e,time:%.2f'%
            (it,loss.item(),inset.loss.item(),bdset.loss.item(),ela))
    t1 = time.time() - t0
    u_e = L1_error(pred_u(netu,inset.X),inset.uu)
    v_e = L1_error(pred_v(netv,inset.X),inset.vv)
    if size > 1:
        if rank > 0:
            comm.send(u_e,dest = 0, tag = 0)
            comm.send(v_e,dest = 0, tag = 1)
        else:
            for rank in range(1,size):
                mau = comm.recv(source = rank, tag = 0)
                mav = comm.recv(source = rank, tag = 1)
                u_e = max(u_e, mau)
                v_e = max(v_e, mav)
            print('use time:%.2f,L1_error of u:%.2e,L1_error of v:%.2e'%(t1,u_e,v_e))
    else:
        print('use time:%.2f,L1_error of u:%.2e,L1_error of v:%.2e'%(t1,u_e,v_e))
def fun_comm(netu,netv,bdset,rank,size,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2):
    if rank == 0:
        comm.send(bdset.X[ind_st1:ind_ed1,:], dest = big, tag = 0)
        x_right = comm.recv(source = big, tag = 0)
        #-----------------------------
        u_right = pred_u(netu,x_right).detach()
        comm.send(u_right, dest = big, tag = 1)
        bdset.uu[ind_st1:ind_ed1,:] = comm.recv(source = big, tag = 1)

        v_right = pred_v(netv,x_right).detach()
        comm.send(v_right, dest = big, tag = 2)
        bdset.vv[ind_st1:ind_ed1,:] = comm.recv(source = big, tag = 2)
    elif rank > 0 and rank < size - 1:
        x_left = comm.recv(source = small, tag = 0)
        comm.send(bdset.X[ind_st2:ind_ed2,:], dest = small, tag = 0)
        #-----------------------------
        u_left = pred_u(netu,x_left).detach()
        bdset.uu[ind_st2:ind_ed2,:] = comm.recv(source = small, tag = 1)
        comm.send(u_left, dest = small, tag = 1)

        v_left = pred_v(netv,x_left).detach()
        bdset.vv[ind_st2:ind_ed2,:] = comm.recv(source = small, tag = 2)
        comm.send(v_left, dest = small, tag = 2)
             
        #qian,hou
        comm.send(bdset.X[ind_st1:ind_ed1,:], dest = big, tag = 0)
        x_right = comm.recv(source = big, tag = 0)
        #-----------------------------
        u_right = pred_u(netu,x_right).detach()
        comm.send(u_right, dest = big, tag = 1)
        bdset.uu[ind_st1:ind_ed1,:] = comm.recv(source = big, tag = 1)

        v_right = pred_v(netv,x_right).detach()
        comm.send(v_right, dest = big, tag = 2)
        bdset.vv[ind_st1:ind_ed1,:] = comm.recv(source = big, tag = 2)
    else:
        x_left = comm.recv(source = small, tag = 0)
        comm.send(bdset.X[ind_st2:ind_ed2,:], dest = small, tag = 0)
        #-----------------------------
        u_left = pred_u(netu,x_left).detach()
        bdset.uu[ind_st2:ind_ed2,:] = comm.recv(source = small, tag = 1)
        comm.send(u_left, dest = small, tag = 1)

        v_left = pred_v(netv,x_left).detach()
        bdset.vv[ind_st2:ind_ed2,:] = comm.recv(source = small, tag = 2)
        comm.send(v_left, dest = small, tag = 2)
def communicate(netu,netv,bdset,rank,size,para):#默认size > 1
    if para == 1:
        big = rank + 1
        small = rank - 1
        ind_st1 = bdset.nx_bd[0]
        ind_ed1 = bdset.nx_bd[0] + bdset.nx_bd[1]
        ind_st2 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
        ind_ed2 = 2*(bdset.nx_bd[0] + bdset.nx_bd[1])
        fun_comm(netu,netv,bdset,rank,size,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)
        
    elif para == 2:
        big = rank + 1
        small = rank - 1
        ind_st1 = bdset.nx_bd[0] + bdset.nx_bd[1]
        ind_ed1 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
        ind_st2 = 0
        ind_ed2 = bdset.nx_bd[0]
        fun_comm(netu,netv,bdset,rank,size,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)
        
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
            bdset.uu[ind_st:ind_ed,:] = comm.recv(source = rank + 1, tag = 1)

            v_right = pred_v(netv,x_right).detach()
            comm.send(v_right, dest = rank + 1, tag = 2)
            bdset.vv[ind_st:ind_ed,:] = comm.recv(source = rank + 1, tag = 2)
            #------
            big = rank + size_x
            small = rank - size_x
            ind_st1 = bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed1 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_st2 = 0
            ind_ed2 = bdset.nx_bd[0]
            fun_comm(netu,netv,bdset,rank_y,size_y,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)
            
        elif rank_x > 0 and rank_x < size_x - 1:
            ind_st = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed = 2*(bdset.nx_bd[0] + bdset.nx_bd[1])
            x_left = comm.recv(source = rank - 1, tag = 0)
            comm.send(bdset.X[ind_st:ind_ed,:], dest = rank - 1, tag = 0)
            #-----------------------------
            u_left = pred_u(netu,x_left).detach()
            bdset.uu[ind_st:ind_ed,:] = comm.recv(source = rank - 1, tag = 1)
            comm.send(u_left, dest = rank - 1, tag = 1)

            v_left = pred_v(netv,x_left).detach()
            bdset.vv[ind_st:ind_ed,:] = comm.recv(source = rank - 1, tag = 2)
            comm.send(v_left, dest = rank - 1, tag = 2)
             
            ind_st = bdset.nx_bd[0]
            ind_ed = bdset.nx_bd[0] + bdset.nx_bd[1]
            comm.send(bdset.X[ind_st:ind_ed,:], dest = rank + 1, tag = 0)
            x_right = comm.recv(source = rank + 1, tag = 0)
            #-----------------------------
            u_right = pred_u(netu,x_right).detach()
            comm.send(u_right, dest = rank + 1, tag = 1)
            bdset.uu[ind_st:ind_ed,:] = comm.recv(source = rank + 1, tag = 1)

            v_right = pred_v(netv,x_right).detach()
            comm.send(v_right, dest = rank + 1, tag = 2)
            bdset.vv[ind_st:ind_ed,:] = comm.recv(source = rank + 1, tag = 2)
            #------
            big = rank + size_x
            small = rank - size_x
            ind_st1 = bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed1 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_st2 = 0
            ind_ed2 = bdset.nx_bd[0]
            fun_comm(netu,netv,bdset,rank_y,size_y,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)
        else:
            ind_st = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed = 2*(bdset.nx_bd[0] + bdset.nx_bd[1])
            x_left = comm.recv(source = rank - 1, tag = 0)
            comm.send(bdset.X[ind_st:ind_ed,:], dest = rank - 1, tag = 0)
            #-----------------------------
            u_left = pred_u(netu,x_left).detach()
            bdset.uu[ind_st:ind_ed,:] = comm.recv(source = rank - 1, tag = 1)
            comm.send(u_left, dest = rank - 1, tag = 1)

            v_left = pred_v(netv,x_left).detach()
            bdset.vv[ind_st:ind_ed,:] = comm.recv(source = rank - 1, tag = 2)
            comm.send(v_left, dest = rank - 1, tag = 2)
            #------
            big = rank + size_x
            small = rank - size_x
            ind_st1 = bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_ed1 = 2*bdset.nx_bd[0] + bdset.nx_bd[1]
            ind_st2 = 0
            ind_ed2 = bdset.nx_bd[0]
            fun_comm(netu,netv,bdset,rank_y,size_y,big,small,ind_st1,ind_ed1,ind_st2,ind_ed2)

              
parser = argparse.ArgumentParser(description='mpi Neural Network Method')
parser.add_argument('--tr', type=int, default=900,
                    help='train size')  
parser.add_argument('--bd', type=int, default=[20,20],
                    help='test size')   
parser.add_argument('--wid', type=int, default=10,
                    help='layers width') 
parser.add_argument('--iter', type=int, default=4,
                    help='max_iter') 
parser.add_argument('--epoch', type=int, default=5,
                    help='initial epoch')                                        
parser.add_argument('--lr', type=float, default=1e0,
                    help='learning rate') 
parser.add_argument('--para', type=int, default=1,
                    help='stragy')
parser.add_argument('--prob', type=int, default=1,
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

layer = [2,lay_wid,lay_wid,1]                  
netu = Net(layer,dtype).to(device)
netv = Net(layer,dtype).to(device)
netp = Net(layer,dtype).to(device)
fnameu = 'size-rank-(%d,%d)unet.pt'%(size,rank)
fnamev = 'size-rank-(%d,%d)vnet.pt'%(size,rank)
fnamep = 'size-rank-(%d,%d)pnet.pt'%(size,rank)
optimtype = 'BFGS'
#optimtype = 'LBFGS'
#optimtype = 'adam'
if optimtype == 'BFGS':
    optim = bfgs.BFGS(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netp.parameters()),lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
elif optimtype == 'LBFGS':
    optim = torch.optim.LBFGS(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netp.parameters()),lr=lr, max_iter=100,
                      tolerance_grad=1e-16, tolerance_change=1e-16,
                      line_search_fn='strong_wolfe')
else:
    optim = torch.optim.Adam(itertools.chain(netu.parameters(),
                                        netv.parameters(),
                                        netp.parameters()),lr=lr)          
inset = INSET(prob,bound,size_tr,size,rank,dtype,device,para)   
bdset = BDSET(prob,bound,nx_bd,size,rank,dtype,device,para)   
#plt.scatter(bdset.X[:,0].detach().numpy(),bdset.X[:,1].detach().numpy()) ;plt.show()
#print(bdset.uu)
start_time = time.time()
for i in range(max_iters):
    
    if rank == 0:
        print('\n    Iters: %d' %(i))
    Train(netu,netv,netp,inset,bdset,optimtype,optim,epoch,rank,size)
    if size > 1:
        communicate(netu,netv,bdset,rank,size,para)
    
torch.save(netu,fnameu)
torch.save(netv,fnamev)
torch.save(netp,fnamep)
ela = time.time() - start_time
netu = netu.to('cpu')
netv = netv.to('cpu')
netv = netv.to('cpu')
nx_te = [21,21]
x_train = np.linspace(bound[0,0],bound[0,1],nx_te[0])
y_train = np.linspace(bound[1,0],bound[1,1],nx_te[1])
inp = torch.zeros(nx_te[0]*nx_te[1],2).type(dtype)
for i in range(nx_te[0]):
    for j in range(nx_te[1]):
        inp[j*nx_te[0] + i,0] = x_train[i]
        inp[j*nx_te[0] + i,1] = y_train[j]

def choose(netu,netv,para,size,rank,inp):
    u_pred = np.zeros_like(inp[:,0:1])
    v_pred = np.zeros_like(inp[:,0:1])
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
    v_pred[ind] = pred_v(netv,inp[ind]).detach().numpy()
    return ind,u_pred,v_pred
ind,u_pred,v_pred = choose(netu,netv,para,size,rank,inp)
if rank > 0:
    comm.send(ind,dest = 0, tag = 0)
    comm.send(u_pred, dest = 0, tag = 1)
    comm.send(v_pred, dest = 0, tag = 2)
else:
    for rank in range(1,size):
        ind = comm.recv(source = rank,tag = 0)
        tmpu = comm.recv(source = rank, tag = 1)
        tmpv = comm.recv(source = rank, tag = 2)
        u_pred[ind] = tmpu[ind]
        v_pred[ind] = tmpv[ind]
    num_line = 40
    cma = 'rainbow'
    x,y = np.meshgrid(x_train,y_train)
    erru = u_pred - UU(inp,[0,0],prob)[:,0:1].detach().numpy()
    errv = v_pred - UU(inp,[0,0],prob)[:,1:2].detach().numpy()
    err = [erru,errv]
    ti = ['u','v']
    fig, ax = plt.subplots(1,2,figsize=(27,18))
    for i in range(2):
        s = ax[i].contourf(x,y,err[i].reshape(nx_te[0],nx_te[1]), num_line, cmap=cma)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig.colorbar(s,ax=ax[i],format = '%.2e')
        
        ax[i].set_title(ti[i])
    plt.savefig('parallel.png')
    plt.show()


