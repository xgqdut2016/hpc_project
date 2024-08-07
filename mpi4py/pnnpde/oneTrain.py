from mpi4py import MPI
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
from scipy.stats import qmc

import bfgs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
np.random.seed(1234)
torch.manual_seed(1234)


def UU(X, order,prob,L):
    x = X[:,0]
    y = X[:,1]
    
    tmp = torch.exp((-x**2 - y**2)/(2*L**2))
    if prob==1:
        if order[0]==0 and order[1]==0:
            return tmp
        if order[0]==1 and order[1]==0:
            return -x*tmp/(L**2)
        if order[0]==0 and order[1]==1:
            return -y*tmp/(L**2)
        if order[0]==2 and order[1]==0:
            return -tmp/(L**2) + tmp*x**2/(L**4)
        if order[0]==0 and order[1]==2:
            return -tmp/(L**2) + tmp*y**2/(L**4)
def FF(X,prob,L):
    return -(UU(X,[2,0],prob,L) + UU(X,[0,2],prob,L))
class INSET():
    def __init__(self,bound,size_tr,prob,L,dtype,dev):
        self.dim = 2
        tmp = self.quasi_samples(size_tr)
        tmp[:,0] = tmp[:,0]*(bound[0,1] - bound[0,0]) + bound[0,0]
        tmp[:,1] = tmp[:,1]*(bound[1,1] - bound[1,0]) + bound[1,0]
        self.x = torch.tensor(tmp).type(dtype)
        self.u_acc = UU(self.x,[0,0],prob,L).reshape(-1,1).type(dtype)
        self.ff = FF(self.x,prob,L).reshape(-1,1).type(dtype)
        
        self.x.requires_grad = True
        self.x = self.x.to(dev)
        self.u_acc = self.u_acc.to(dev)
        self.ff = self.ff.to(dev)
        self.weight_grad = torch.ones_like(self.ff).to(dev)
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample
class BDSET():#边界点取值
    def __init__(self,bound,nx,prob,L,dtype,dev):
        self.dim = 2
        self.DS = 2*(nx[0] + nx[1])
        
        self.hx = [(bound[0,1] - bound[0,0])/nx[0],(bound[1,1] - bound[1,0])/nx[1]]
        
        self.x = torch.zeros(self.DS,2)#储存内点
        m = 0
        for i in range(nx[0]):
            self.x[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
            self.x[m,1] = bound[1,0] 
            m = m + 1
        for j in range(nx[1]):
            self.x[m,0] = bound[0,1]
            self.x[m,1] = bound[1,0] + (j + 0.5)*self.hx[1]
            m = m + 1
        for i in range(nx[0]):
            self.x[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
            self.x[m,1] = bound[1,1] 
            m = m + 1
        for j in range(nx[1]):
            self.x[m,0] = bound[0,0]
            self.x[m,1] = bound[1,0] + (j + 0.5)*self.hx[1]
            m = m + 1
        self.Dright = UU(self.x,[0,0],prob,L).reshape(-1,1)
        self.x = self.x.type(dtype)
        self.Dright = self.Dright.type(dtype)
        
        self.x = self.x.to(dev)
        self.Dright = self.Dright.to(dev)
        
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
def pred_u(netf,X):
    return netf.forward(X)


    
def error(u_pred, u_acc):
    u_pred = u_pred.to(device)
    u_acc = u_acc.to(device)
    #return (((u_pred-u_acc)**2).sum()/(u_acc**2).sum()) ** (0.5)
    #return (((u_pred-u_acc)**2).mean()) ** (0.5)
    return max(abs(u_pred - u_acc))
def Lossf(netf,inset,bdset):
    if inset.x.requires_grad is not True:
        inset.x.requires_grad = True
    
    inset.u = pred_u(netf,inset.x)
    u_x, = torch.autograd.grad(inset.u, inset.x, create_graph=True, retain_graph=True,
                                   grad_outputs=inset.weight_grad)
    
    
    u_xx, = torch.autograd.grad(u_x[:,0:1], inset.x, create_graph=True, retain_graph=True,
                                  grad_outputs=inset.weight_grad)
    u_yy, = torch.autograd.grad(u_x[:,1:2], inset.x, create_graph=True, retain_graph=True,
                                  grad_outputs=inset.weight_grad)
    out_in = ((u_xx[:,0:1] + u_yy[:,1:2] + inset.ff)**2).mean()
    
    beta = 1e0
    ub = pred_u(netf,bdset.x)
    out_b = ((ub - bdset.Dright)**2).mean()
    
    res = out_in + beta*out_b
    return torch.sqrt(res)
def train(netf,inset,bdset,optim,epoch,rank):
    print('Train u Neural Network')
    loss = Lossf(netf,inset,bdset)
    
    for it in range(epoch):
        st = time.time()
        def closure():
            loss = Lossf(netf,inset,bdset)
            optim.zero_grad()
            loss.backward()
            return loss
        optim.step(closure) 
        loss = Lossf(netf,inset,bdset)
        err = error(inset.u,inset.u_acc)
        ela = time.time() - st
        print('rank:%d,epoch: %d, loss: %.3e,L1_err:%.3e, time: %.2f'
              %(rank,(it+1), loss.item(),err, ela))
size_tr = 100
nx_bd = [20,20]
bound = np.array([0,1,0,1]).reshape(2,2)
prob = 1
dtype = torch.float64
wid = 10
#----------------------
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
ls = np.linspace(0.1,1,size)

L = ls[rank]
inset = INSET(bound,size_tr,prob,L,dtype,device)
bdset = BDSET(bound,nx_bd,prob,L,dtype,device)
layers = [2,wid,wid,1];netf = Net(layers,dtype).to(device)
tests_num = 1
lr_f = 1e0
epoch = 5

filename = 'lay%dL%du.pt'%(wid,rank)
for it in range(tests_num):
    optim = bfgs.BFGS(netf.parameters(), 
                      lr=lr_f, max_iter = 100,
                      tolerance_grad=1e-15, tolerance_change=1e-15,
                      line_search_fn='strong_wolfe')

    st = time.time()
    train(netf,inset,bdset,optim,epoch,rank)
    ela = time.time() - st
torch.save(netf,filename)

    

fig, ax = plt.subplots(1,3,figsize=(18,3))

n = 64
for j in range(3):
    ax[j].axis([bound[0,0],bound[0,1],bound[1,0],bound[1,1]])
    ax[j].set_xlim([bound[0,0],bound[0,1]])
    ax[j].set_ylim([bound[1,0],bound[1,1]])
    ax[j].axis('off')

num_line = 100
x_train = np.linspace(bound[0,0],bound[0,1],n)
y_train = np.linspace(bound[1,0],bound[1,1],n)
xx,yy = np.meshgrid(x_train,y_train)
x0 = xx.reshape(-1,1)
x1 = yy.reshape(-1,1)
te_data_tra = torch.from_numpy(np.hstack([x0,x1]))
te_data = torch.zeros([te_data_tra.shape[0],2]).type(dtype)
te_data[:,0:2] = te_data_tra

net_u = netf.to('cpu')


u = pred_u(net_u, te_data).cpu().detach().numpy().flatten()
u_acc = UU(te_data,[0,0],prob,L).detach().numpy().flatten()


x0 = x0.flatten()
x1 = x1.flatten()

ax00 = ax[0].tricontourf(x0, x1, u, num_line, cmap='rainbow')

fig.colorbar(ax00,ax=ax[0],fraction = 0.03,pad = 0.01)
ax[0].set_title('L:%.2f,PINN u'%(L),fontsize=15)

ax01 = ax[1].tricontourf(x0, x1, u_acc, num_line, cmap='rainbow')

fig.colorbar(ax01,ax=ax[1],fraction = 0.03,pad = 0.01)
ax[1].set_title('L:%.2f,u_acc'%(L),fontsize=15)

ax02 = ax[2].tricontourf(x0, x1, u - u_acc, num_line, cmap='rainbow')

fig.colorbar(ax02,ax=ax[2],fraction = 0.03,pad = 0.01)
ax[2].set_title('L:%.2f,err'%(L),fontsize=15)
fig.tight_layout()
plt.show()





