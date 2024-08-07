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
#bound = [t_a,t_b,x_a,x_b].reshape(2,2)
class INSET():
    def __init__(self,bound,size_tr,size,rank,dtype,dev):
        self.dim = bound.shape[0]
        tmp = self.quasi_samples(size_tr)
        tmp[:,0] = tmp[:,0]*(bound[0,1] - bound[0,0]) + bound[0,0]
        tmp[:,1] = tmp[:,1]*(bound[1,1] - bound[1,0]) + bound[1,0]
        self.lap_num = 4
        if size == 1:
            self.X = torch.tensor(tmp).to(dtype)
        else:
            self.hl = (bound[1,1] - bound[1,0])/size
            self.lap = self.hl/self.lap_num
            if rank == 0:
                sub_domain = [bound[1,0] + rank*self.hl,bound[1,0] + (rank + 1)*self.hl + self.lap]
            elif rank == size - 1:
                sub_domain = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl]
            else:
                sub_domain = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl + self.lap]
            ind = (tmp[:,1] >= sub_domain[0])*(tmp[:,1] <= sub_domain[1])
            
            self.X = torch.tensor(tmp[ind]).to(dtype)
        
        self.X = self.X.type(dtype)
        self.X = self.X.to(dev).data
        
        self.weight_grad = torch.ones_like(self.X[:,0:1]).to(dev)
    def quasi_samples(self,N):
        sampler = qmc.Sobol(d=self.dim)
        sample = sampler.random(n=N)
        return sample
        

class BDSET():
    def __init__(self,bound,nx_bd,size,rank,dtype,dev):
        self.dim = bound.shape[0]
        self.lap_num = 4
        self.nx_bd = nx_bd
        self.X = torch.zeros(2*nx_bd[0] + nx_bd[1],2)
        if size == 1:
            self.hx = [(bound[0,1] - bound[0,0])/nx_bd[0],(bound[1,1] - bound[1,0])/nx_bd[1]]
            m = 0
            for i in range(nx_bd[0]):#t,x = -1
                self.X[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
                self.X[m,1] = bound[1,0]
                m += 1
            for i in range(nx_bd[0]):#t,x = 1
                self.X[m,0] = bound[0,0] + (i + 0.5)*self.hx[0]
                self.X[m,1] = bound[1,1]
                m += 1 
            for j in range(nx_bd[1]):#t = 0,x
                self.X[m,0] = bound[0,0]
                self.X[m,1] = bound[1,0] + (j + 0.5)*self.hx[1]
                m += 1 
        else:
            self.hl = (bound[1,1] - bound[1,0])/size
            self.lap = self.hl/self.lap_num
            if rank == 0:
                sub_domain_y = [bound[1,0] + rank*self.hl,bound[1,0] + (rank + 1)*self.hl + self.lap]
            elif rank == size - 1:
                sub_domain_y = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl]
            else:
                sub_domain_y = [bound[1,0] + rank*self.hl - self.lap,bound[1,0] + (rank + 1)*self.hl + self.lap]
            sub_domain_x = [bound[0,0],bound[0,1]]
            
                
            self.hx = [(sub_domain_x[1] - sub_domain_x[0])/nx_bd[0],(sub_domain_y[1] - sub_domain_y[0])/nx_bd[1]]
            m = 0
            for i in range(nx_bd[0]):
                self.X[m,0] = sub_domain_x[0] + (i + 0.5)*self.hx[0]
                self.X[m,1] = sub_domain_y[0]
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
        ind_t = (self.X[:,0] >= bound[0,0])*(self.X[:,0] < bound[0,0] + eps)
        ind_xa = (self.X[:,1] >= bound[1,0])*(self.X[:,1] < bound[1,0] + eps)
        ind_xb = (self.X[:,1] > bound[1,1] - eps)*(self.X[:,1] <= bound[1,1])
        
        self.X = self.X.type(dtype)
        self.X = self.X.to(dev)
        #plt.scatter(self.X[ind,0].detach().numpy(),self.X[ind,1].detach().numpy());plt.show()
        self.u_acc = torch.zeros_like(self.X[:,0:1])
        self.u_acc[ind_t] = -torch.sin(np.pi*self.X[ind_t][:,1:2]).reshape(-1,1)
        self.u_acc[ind_xa] = 0
        self.u_acc[ind_xb] = 0
        
        
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
    return netu.forward(X)#*(X[:,1:2]**2 - 1)

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
    u_t = ux[:,0:1]
    u_x = ux[:,1:2]
    u_xx, = torch.autograd.grad(u_x, inset.X, create_graph=True, retain_graph=True,
                                 grad_outputs=inset.weight_grad)
    
    inset.res_u = (u_t + inset.u*u_x - 0.01*u_xx[:,1:2]/np.pi)**2
    inset.pde_u = torch.sqrt(inset.res_u).mean()
    bdset.res_u = (bdset.u - bdset.u_acc.data)**2
    bdset.pde_u = torch.sqrt(bdset.res_u).mean()
    return inset.pde_u + bdset.pde_u
    #return inset.pde_u
def Train(netu,inset,bdset,optimtype,optim,epoch,rank):
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
    if rank == 0:
        print('use time:%.2f'%(t1))
def communicate(netu,bdset,rank):
    ind_st1 = 0
    ind_ed1 = bdset.nx_bd[0]

    ind_st2 = bdset.nx_bd[0]
    ind_ed2 = 2*bdset.nx_bd[0]
    if rank == 0:
        
        comm.send(bdset.X[ind_st1:ind_ed1,:], dest = rank + 1, tag = 0)
        x_right = comm.recv(source = rank + 1, tag = 0)
        #-----------------------------
        u_right = pred_u(netu,x_right).detach()
        comm.send(u_right, dest = rank + 1, tag = 1)
        bdset.u_acc[ind_st1:ind_ed1,:] = comm.recv(source = rank + 1, tag = 1)
    elif rank > 0 and rank < size - 1:
        x_left = comm.recv(source = rank - 1, tag = 0)
        comm.send(bdset.X[ind_st2:ind_ed2,:], dest = rank - 1, tag = 0)
        #-----------------------------
        u_left = pred_u(netu,x_left).detach()
        bdset.u_acc[ind_st2:ind_ed2,:] = comm.recv(source = rank - 1, tag = 1)
        comm.send(u_left, dest = rank - 1, tag = 1)
             
        #qian,hou
        comm.send(bdset.X[ind_st1:ind_ed1,:], dest = rank + 1, tag = 0)
        x_right = comm.recv(source = rank + 1, tag = 0)
        #-----------------------------
        u_right = pred_u(netu,x_right).detach()
        comm.send(u_right, dest = rank + 1, tag = 1)
        bdset.u_acc[ind_st1:ind_ed1,:] = comm.recv(source = rank + 1, tag = 1)
    else:
        x_left = comm.recv(source = rank - 1, tag = 0)
        comm.send(bdset.X[ind_st2:ind_ed2,:], dest = rank - 1, tag = 0)
        #-----------------------------
        u_left = pred_u(netu,x_left).detach()
        bdset.u_acc[ind_st2:ind_ed2,:] = comm.recv(source = rank - 1, tag = 1)
        comm.send(u_left, dest = rank - 1, tag = 1)
        
bound = np.array([0,0.99,-1,1]).reshape(2,2)               
parser = argparse.ArgumentParser(description='mpi Neural Network Method')
parser.add_argument('--tr', type=int, default=800,
                    help='train size')  
parser.add_argument('--bd', type=int, default=[20,20],
                    help='test size')   
parser.add_argument('--wid', type=int, default=10,
                    help='layers width') 
parser.add_argument('--iter', type=int, default=10,
                    help='max_iter') 
parser.add_argument('--epoch', type=int, default=2,
                    help='initial epoch')                                        
parser.add_argument('--lr', type=float, default=1e0,
                    help='learning rate') 
                                         
dtype = torch.float64
args = parser.parse_args()
size_tr = args.tr
nx_bd = args.bd
lay_wid = args.wid
max_iters = args.iter
epoch = args.epoch  
lr = args.lr


layer_u = [2,lay_wid,lay_wid,1]                  
netu = Net(layer_u,dtype).to(device)
fname = 'size-rank-(%d,%d)unet.pt'%(size,rank)
optimtype = 'BFGS'
#optimtype = 'LBFGS'
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
inset = INSET(bound,size_tr,size,rank,dtype,device)   
bdset = BDSET(bound,nx_bd,size,rank,dtype,device)   
#print(bdset.u_acc)
#plt.scatter(bdset.X[:,0].detach().numpy(),bdset.X[:,1].detach().numpy()) ;plt.show()
#print(Loss_fun(netu,inset,bdset))
start_time = time.time()
for i in range(max_iters):
    
    if rank == 0:
        print('\n    Iters: %d' %(i))
    Train(netu,inset,bdset,optimtype,optim,epoch,rank)
    if size > 1:
        communicate(netu,bdset,rank)
    epoch += 1
    

ela = time.time() - start_time
nx_te = [21,21]
x_train = np.linspace(bound[0,0],bound[0,1],nx_te[0])
y_train = np.linspace(bound[1,0],bound[1,1],nx_te[1])
inp = torch.zeros(nx_te[0]*nx_te[1],2).type(dtype)
for i in range(nx_te[0]):
    for j in range(nx_te[1]):
        inp[j*nx_te[0] + i,0] = x_train[i]
        inp[j*nx_te[0] + i,1] = y_train[j]

def choose(netu,size,rank,inp):
    u_pred = np.zeros_like(inp[:,0:1])
    
   
    hl = (bound[1,1] - bound[1,0])/size
    sub_domain = [bound[1,0] + rank*hl,bound[1,0] + (rank + 1)*hl]
    ind = (inp[:,1] >= sub_domain[0])*(inp[:,1] <= sub_domain[1])
    u_pred[ind] = pred_u(netu,inp[ind]).detach().numpy()
    return ind,u_pred
ind,u_pred = choose(netu,size,rank,inp)
if size == 1:
    np.save('burger.npy',u_pred)
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
    err = u_pred.reshape(nx_te[0],nx_te[1])
    fig, ax = plt.subplots(figsize=(27,18))
    s = ax.contourf(x,y,err, num_line, cmap=cma)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    fig.colorbar(s,ax=ax,format = '%.2e')
    ax.set_xlabel('x',fontsize = 25)
    ax.set_ylabel('y',fontsize = 25)
    ax.set_title('burgers',fontsize = 25)
    plt.savefig('parallel.png')
    plt.show()


