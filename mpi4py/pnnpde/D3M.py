from mpi4py import MPI
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
from scipy.stats import qmc
import argparse
import bfgs 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
np.random.seed(1234)
torch.manual_seed(1234)

a_min = 0
a_max = 1
b_min = 0
b_max = 1
L_min = 0.1
L_max = 1
bound = np.array([0,1,0,1,a_min,a_max,b_min,b_max,L_min,L_max]).reshape(5,2)

def UU(X, order,prob):
    x = X[:,0]
    y = X[:,1]
    a = X[:,2]
    b = X[:,3]
    L = X[:,4]
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
def FF(X,prob):
    return -(UU(X,[2,0],prob) + UU(X,[0,2],prob))
class INSET():
    def __init__(self,bound,size_tr,prob,size,rank,dtype,dev):
        self.dim = bound.shape[0]
        self.hl = (bound[4,1] - bound[4,0])/size
        tmp = self.quasi_samples(size_tr)
        #print(type(tmp),type(bound))
        tmp[:,0] = tmp[:,0]*(bound[0,1] - bound[0,0]) + bound[0,0]
        tmp[:,1] = tmp[:,1]*(bound[1,1] - bound[1,0]) + bound[1,0]
        tmp[:,2] = tmp[:,2]*(bound[2,1] - bound[2,0]) + bound[2,0]
        tmp[:,3] = tmp[:,3]*(bound[3,1] - bound[3,0]) + bound[3,0]
        tmp[:,4] = tmp[:,4]*self.hl + bound[4,0] + rank*self.hl
        self.x = torch.tensor(tmp).to(dtype)
        self.u_acc = UU(self.x,[0,0],prob).reshape(-1,1).to(dtype)
        self.ff = FF(self.x,prob).reshape(-1,1).to(dtype)
        
        self.x.requires_grad = True
        self.x = self.x.to(dev)
        self.u_acc = self.u_acc.to(dev)
        self.ff = self.ff.to(dev)
        self.weight_grad = torch.ones_like(self.ff).to(dev).to(dtype)
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample
class BDSET():#边界点取值
    def __init__(self,bound,nx,prob,size,rank,dtype,dev):
        self.dim = bound.shape[0]
        self.size = 2*(nx[0] + nx[1])
        self.hl = (bound[4,1] - bound[4,0])/size
        tmp = self.quasi_samples(self.size)
        self.hx = [(bound[0,1] - bound[0,0])/nx[0],(bound[1,1] - bound[1,0])/nx[1]]
        tmp[:,2] = tmp[:,2]*(bound[2,1] - bound[2,0]) + bound[2,0]
        tmp[:,3] = tmp[:,3]*(bound[3,1] - bound[3,0]) + bound[3,0]
        tmp[:,4] = tmp[:,4]*self.hl + bound[4,0] + rank*self.hl
        self.x = torch.tensor(tmp).to(dtype)
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
        self.u_acc = UU(self.x,[0,0],prob).reshape(-1,1).to(dtype)
        
        self.x = self.x.to(dev)
        self.u_acc = self.u_acc.to(dev)
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample
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
    out_b = ((ub - bdset.u_acc)**2).mean()
    
    res = out_in + beta*out_b
    return torch.sqrt(res)
def train(netf,inset,bdset,optim,epoch,rank):
    if (rank == 0):
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
        if (rank == 0):
            print('rank:%d,epoch: %d, loss: %.3e,L1_err:%.3e, time: %.2f'
              %(rank,(it+1), loss.item(),err, ela))
parser = argparse.ArgumentParser(description='mpi Neural Network Method')
parser.add_argument('--tr', type=int, default=500,
                    help='train size')  
parser.add_argument('--nx_bd', type=int, default=[16,16],
                    help='test size')   
parser.add_argument('--prob', type=int, default=1,
                    help='problem id')        
parser.add_argument('--wid', type=int, default=10,
                    help='layers width') 
parser.add_argument('--epoch', type=int, default=10,
                    help='max_iter')  
parser.add_argument('--lr', type=float, default=1e0,
                    help='learning rate')                                                          
dtype = torch.float64
args = parser.parse_args()
size_tr = args.tr
nx_bd = args.nx_bd
prob = args.prob
wid = args.wid
layers = [5,wid,wid,1];netf = Net(layers,dtype).to(device)
epoch = args.epoch
lr_f = args.lr
#----------------------

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
inset = INSET(bound,size_tr,prob,size,rank,dtype,device)
bdset = BDSET(bound,nx_bd,prob,size,rank,dtype,device)

tests_num = 1



filename = 'lay-size-rank-(%d,%d,%d)u.pt'%(wid,size,rank)
for it in range(tests_num):
    optim = bfgs.BFGS(netf.parameters(), 
                      lr=lr_f, max_iter = 100,
                      tolerance_grad=1e-15, tolerance_change=1e-15,
                      line_search_fn='strong_wolfe')

    st = time.time()
    train(netf,inset,bdset,optim,epoch,rank)
    ela = time.time() - st
torch.save(netf,filename)
number = 8
nx_te_in = [64,64,number]
x_train = np.linspace(bound[0,0],bound[0,1],nx_te_in[0])
y_train = np.linspace(bound[1,0],bound[1,1],nx_te_in[1])
L = np.linspace(bound[4,0],bound[4,1],nx_te_in[2])
a = 0.5
b = 0.5
xx = torch.zeros(nx_te_in[0]*nx_te_in[1]*number,5)
m = 0
for i in range(nx_te_in[0]):
    for j in range(nx_te_in[1]):
        for k in range(number):
            xx[m,0] = x_train[i]
            xx[m,1] = y_train[j]
            xx[m,4] = L[k]
            xx[m,2] = a
            xx[m,3] = b
            m = m + 1

hl = (bound[4,1] - bound[4,0])/size
sub_domain = torch.tensor([bound[4,0] + hl*rank,bound[4,0] + hl*(rank + 1)])
ind = (xx[:,4] >= sub_domain[0])*(xx[:,4] <= sub_domain[1])
sub_x = xx[ind].type(dtype)
Ua = UU(sub_x,[0,0],prob).detach().numpy().reshape(-1,1)
netf = netf.to('cpu')
U = pred_u(netf,sub_x).detach().numpy()
if rank > 0: 
    comm.send(ind,dest = 0, tag = 0)
    comm.send(Ua,dest = 0, tag = 1)
    comm.send(U,dest = 0, tag = 2)
    
else:
    u_pred = np.zeros([nx_te_in[0]*nx_te_in[1]*number,1])
    u_acc = np.zeros([nx_te_in[0]*nx_te_in[1]*number,1])
    
    u_pred[ind] = U
    u_acc[ind] = Ua
    for rank in range(1,size):
        ind = comm.recv(source = rank,tag = 0)
        Ua = comm.recv(source = rank, tag = 1)
        U = comm.recv(source = rank, tag = 2)
        
        u_pred[ind] = U
        u_acc[ind] = Ua

    frac = 0.02
    padding = 0.2
    fig, ax = plt.subplots(3,number,figsize=(27,9))
    for i in range(3):
        for j in range(number):
            ax[i,j].axis('equal')
            ax[i,j].set_xlim([bound[0,0],bound[0,1]])
            ax[i,j].set_ylim([bound[1,0],bound[1,1]])
            ax[i,j].axis('off')
        
    num_line = 100

    x0, x1 = np.meshgrid(x_train,y_train)
    u_pred = u_pred.reshape(-1,number)
    u_acc = u_acc.reshape(-1,number)
    for i in range(number):
        u = u_pred[:,i].reshape(nx_te_in[0],nx_te_in[1]).T
        ua = u_acc[:,i].reshape(nx_te_in[0],nx_te_in[1]).T
    
        s0 = ax[0,i].contourf(x0, x1, u, num_line, cmap='rainbow')
        ax[0,i].contour(s0, linewidths=0.6, colors='black')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        ax[0,i].set_title('L = %.2f'%(L[i]),fontsize=15)    
        fig.colorbar(s0,ax=ax[0,i])

        s1 = ax[1,i].contourf(x0, x1, ua, num_line, cmap='rainbow')
        ax[1,i].contour(s1, linewidths=0.6, colors='black')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig.colorbar(s1,ax=ax[1,i])

        s2 = ax[2,i].contourf(x0, x1, u-ua, num_line, cmap='rainbow')
        ax[2,i].contour(s2, linewidths=0.6, colors='black')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        fig.colorbar(s2,ax=ax[2,i],format = '%.00e')
         
    
    plt.suptitle('(a,b) = (%.2f,%.2f)'%(a,b))
    ax[0,0].text(-0.5,0.5,'parallel:',fontsize=15)
    ax[1,0].text(-0.5,0.5,'Exact:',fontsize=15)
    ax[2,0].text(-0.5,0.5,'err:',fontsize=15)
    plt.savefig('gtparallel.png')
    fig.tight_layout()
    plt.show()
    para_num = size*netf.total_para()
    tr_num = size*size_tr
    bd_num = size*(2*nx_bd[0] + 2*nx_bd[1])
    print('total para:%d, size_tr:%d,bd num:%d\n'%(para_num,tr_num,bd_num))


