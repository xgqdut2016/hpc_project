import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
from scipy.stats import qmc
wid = 10
device = 'cpu'
a = 0
b = 0
def UU(X, order,prob,L):
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
def phi(x):
    ind0 = (x[:,0] > -1)*(x[:,0] <= 0);ind1 = (x[:,0] > 0)*(x[:,0] <= 1)
    return (1 + x[:,0])*ind0 + (1 - x[:,0])*ind1
def wise(x,i,ls):
    hx = ls[1] - ls[0]
    tmp = (x - ls[i])/hx
    return phi(tmp).reshape(-1,1)
def gauss(x,node,ls):
    uh = torch.zeros_like(x[:,0])
    for i in range(len(ls)):
        uh += node[i]*wise(x,i,ls)
    return uh.reshape(-1,1)
def pred_gauss(x,ls,L):
    
    uh = 0*x[:,0:1]
    for i in range(len(ls)):
        netf = torch.load('lay%dL%du.pt'%(wid,i)).to(device)
        #print(wise(x,i,ls).shape,pred_u(netf,x).shape)
        
        uh += wise(L,i,ls)*pred_u(netf,x)
    return uh.reshape(-1,1)
size = 4

bound = np.array([0,1,0,1]).reshape(2,2)
dtype = torch.float64

prob = 1
ls = np.linspace(0.1,1,size)
tmp = 0.6
L = np.array([[tmp]])
L = torch.tensor(L)

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


u = pred_gauss(te_data,ls,L).cpu().detach().numpy().flatten()
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
plt.savefig('L.png')
plt.show()






