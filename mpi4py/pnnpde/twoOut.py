from mpi4py import MPI
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
from scipy.stats import qmc




device = torch.device('cpu')
np.random.seed(1234)
torch.manual_seed(1234)
L = 1.0
wid = 8

def UU(X, order,prob,a,b):
    x = X[:,0]
    y = X[:,1]
    
    tmp = torch.exp((-(x - a)**2 - (y - b)**2)/(2*L**2))
    if prob==1:
        if order[0]==0 and order[1]==0:
            return tmp
        if order[0]==1 and order[1]==0:
            return -(x - a)*tmp/(L**2)
        if order[0]==0 and order[1]==1:
            return -(y - b)*tmp/(L**2)
        if order[0]==2 and order[1]==0:
            return -tmp/(L**2) + tmp*(x - a)**2/(L**4)
        if order[0]==0 and order[1]==2:
            return -tmp/(L**2) + tmp*(y - b)**2/(L**4)
def FF(X,prob,a,b):
    return -(UU(X,[2,0],prob,a,b) + UU(X,[0,2],prob,a,b))

        
    
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
class FENET():
    def __init__(self,nx,bound):
        self.dim = 2
        self.hx = [(bound[0,1] - bound[0,0])/(nx[0] - 1),(bound[1,1] - bound[1,0])/(nx[1] - 1)]
        self.bound = bound
        
        self.nx = nx
        self.area = (bound[0,1] - bound[0,0])*(bound[1,1] - bound[1,0])
        
        
        self.Nodes_create()
        
    def Nodes_create(self):
        self.Nodes_size = self.nx[0]*self.nx[1]
        self.Node = np.zeros([self.Nodes_size,self.dim])
        for j in range(self.nx[1]):
            for i in range(self.nx[0]):
                self.Node[j*self.nx[0] + i,0] = self.bound[0,0] + i*self.hx[0]
                self.Node[j*self.nx[0] + i,1] = self.bound[1,0] + j*self.hx[1]
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
        temp = (X - self.Node[i,:])/np.array([self.hx[0],self.hx[1]])
        if order == [0,0]:
            return self.phi(temp,order)
        if order == [1,0]:
            return self.phi(temp,order)/self.hx[0]
        if order == [0,1]:
            return self.phi(temp,order)/self.hx[1]
    
    def solve(self,mu,x):#mu = [1,2]
        uh = 0*x[:,0:1]
        for j in range(self.nx[1]):
            for i in range(self.nx[0]):
                filename = '(lay,i,j)-(%d,%d,%d)u.pt'%(wid,i,j)
                netf = torch.load(filename).to(device)
                uh += self.basic(mu,[0,0],j*self.nx[0] + i)*pred_u(netf,x)

        return uh
def error(u_pred, u_acc):
    u_pred = u_pred.to(device)
    u_acc = u_acc.to(device)
    #return (((u_pred-u_acc)**2).sum()/(u_acc**2).sum()) ** (0.5)
    #return (((u_pred-u_acc)**2).mean()) ** (0.5)
    return max(abs(u_pred - u_acc))

size = 10
M = 10
N = size
dtype = torch.float64
bound = np.array([0,1,0,1]).reshape(2,2)

prob = 1
abbound = np.array([0,1,0,1]).reshape(2,2)


nx = [M,N]
fe = FENET(nx,abbound)

a = 0.031
b = 0.781
mu = np.array([[a,b]])
mu = torch.tensor(mu)
print(mu.shape)
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

u = fe.solve(mu,te_data).cpu().detach().numpy().flatten()
u_acc = UU(te_data,[0,0],prob,a,b).detach().numpy().flatten()


x0 = x0.flatten()
x1 = x1.flatten()

ax00 = ax[0].tricontourf(x0, x1, u, num_line, cmap='rainbow')

fig.colorbar(ax00,ax=ax[0],fraction = 0.03,pad = 0.01)
ax[0].set_title('(a,b) = (%.2f,%.2f),PINN u'%(a,b),fontsize=15)

ax01 = ax[1].tricontourf(x0, x1, u_acc, num_line, cmap='rainbow')

fig.colorbar(ax01,ax=ax[1],fraction = 0.03,pad = 0.01)
ax[1].set_title('(a,b) = (%.2f,%.2f),u_acc'%(a,b),fontsize=15)

ax02 = ax[2].tricontourf(x0, x1, u - u_acc, num_line, cmap='rainbow')

fig.colorbar(ax02,ax=ax[2],fraction = 0.03,pad = 0.01)
ax[2].set_title('(a,b) = (%.2f,%.2f),err'%(a,b),fontsize=15)
fig.tight_layout()
plt.show()





