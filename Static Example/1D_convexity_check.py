#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:45:13 2024

@author: jarrah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:08:19 2024

@author: jarrah
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
from torch.distributions.multivariate_normal import MultivariateNormal
import sys

plt.close('all')
plt.rc('font', size=13)          # controls default text sizes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


torch.manual_seed(333)
np.random.seed(101)
class f_nn(torch.nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim):
        super(f_nn, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.activationSigmoid = nn.Sigmoid()
        self.activationReLu = nn.ReLU()#ReLU()
        self.layer_input = nn.Linear(self.x_dim + self.y_dim, self.hidden_dim, bias=True)
        self.layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layerout = nn.Linear(self.hidden_dim, 1, bias=False)
            
    def forward(self, x,y):
        h = self.layer_input(torch.concat((x,y),dim=1))
        h_temp = self.layer_1(self.activationReLu(h)) 
        z = self.layerout(self.activationReLu(h_temp) + h)  #+ 0.01*(x*x).sum(dim=1)
        return z
# =============================================================================
# class f_nn(torch.nn.Module):
#     def __init__(self, x_dim, y_dim, hidden_dim):
#         super(f_nn, self).__init__()
#         self.x_dim = x_dim
#         self.y_dim = y_dim
#         self.hidden_dim = hidden_dim
#         self.activationSigmoid = nn.Sigmoid()
#         self.activationReLu = nn.ReLU()
#         self.layer_input = nn.Linear(self.x_dim + self.y_dim, self.hidden_dim, bias=True)
#         self.layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#         
#         self.layer_2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#         
#         self.layerout = nn.Linear(self.hidden_dim, 1, bias=False)
#             
#     def forward(self, x,y):
#         h = self.layer_input(torch.concat((x,y),dim=1))
#         h_temp = self.layer_1(self.activationReLu(h)) 
#         
#         h_temp = self.layer_2(self.activationReLu(h_temp) + h) 
#         z = self.layerout(self.activationReLu(h_temp) + h)  #+ 0.01*(x*x).sum(dim=1)
#         
#         return z
# =============================================================================
# =============================================================================
# class T_map(torch.nn.Module):
#     def __init__(self, x_dim, y_dim, hidden_dim):
#         super(T_map, self).__init__()
#         self.x_dim = x_dim
#         self.y_dim = y_dim
#         self.hidden_dim = hidden_dim
#         self.activationSigmoid = nn.Sigmoid()
#         self.activationReLu = nn.ReLU()
#         self.layer_input = nn.Linear(self.x_dim + self.y_dim, self.hidden_dim, bias=True)
#         self.layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
#         self.layerout = nn.Linear(self.hidden_dim, x_dim, bias=True)
#                 
#     def forward(self, x,y):
#         h = self.layer_input(torch.concat((x,y),dim=1))
#         h_temp = self.layer_1(self.activationReLu(h)) 
#         z = self.layerout(self.activationReLu(h_temp) + h) 
#         return 10*nn.Tanh()(z)
# =============================================================================

class T_map(nn.Module):
    
    def __init__(self, x_dim, y_dim, hidden_dim):
        super(T_map, self).__init__()
        self.input_dim = [0,0]
        self.input_dim[0] = x_dim
        self.input_dim[1] = y_dim
        self.hidden_dim = hidden_dim
        self.activationSigmoid = nn.Sigmoid()
        self.activationReLu = nn.ReLU()
        self.activationNonLinear = nn.Sigmoid()
        self.layer_input = nn.Linear(self.input_dim[0]+self.input_dim[1], self.hidden_dim, bias=False)
        self.layer_x = nn.Linear(self.input_dim[0], self.hidden_dim, bias=False)
        self.layer_y = nn.Linear(self.input_dim[1], self.hidden_dim, bias=False)
        self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layerout = nn.Linear(self.hidden_dim, self.input_dim[0], bias=False)
        
        self.A = torch.nn.Parameter(torch.randn(self.input_dim[0],self.input_dim[0]))
        self.m_hat = torch.nn.Parameter(torch.randn(self.input_dim[0]))
        self.o_hat = torch.nn.Parameter(torch.randn(self.input_dim[1]))
        self.K = torch.nn.Parameter(torch.randn(self.input_dim[1],self.input_dim[0]))
        
        self.dist = MultivariateNormal(torch.zeros(self.input_dim[1]),gamma*gamma * torch.eye(self.input_dim[1]))
    # Input is of size
    def forward(self, x,y):
        
# =============================================================================
#         eta = self.dist.sample((x.shape[0],))         
#         m_hat = x.T.mean(axis=1)
#         o_hat = (h(x.T)).mean(axis=1)
#         a = (x - m_hat)
#         b = (h(x.T).T - o_hat)
#         C_hat_vh = (a.T@b)/x.shape[0]
#         C_hat_hh = (b.T@b)/x.shape[0]
#         K = C_hat_vh @ torch.linalg.inv(C_hat_hh + torch.eye(self.input_dim[1])*gamma*gamma)
#         x = x + (K@ (y - h(x.T).T - eta).T).T 
# =============================================================================  
        
        X = self.layer_input(torch.concat((x,y),dim=1))
        
        xy = self.layer11(X)
        xy = self.activationReLu(xy)
        xy = self.layer12 (xy)
        xy = self.activationReLu(xy)
        
# =============================================================================
#         xy = self.layer21(xy + X)
#         xy = self.activationReLu(xy)
#         xy = self.layer22 (xy)
#         xy = self.activationReLu(xy)
#         xy = self.layerout(xy) + x
# =============================================================================
        xy = self.layerout(xy + X)#+x
        return xy 
    
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.1)

def train(f,T,X,Y,ITERS,LR,BATCH_SIZE):
    f.train()
    T.train()
    optimizer_T = torch.optim.Adam(T.parameters(), lr=LR/1) 
    optimizer_f = torch.optim.Adam(f.parameters(), lr=LR/1)
    scheduler_f = ExponentialLR(optimizer_f, gamma=0.999) #set LR = 1e-1
    scheduler_T = ExponentialLR(optimizer_T, gamma=0.999) #set LR = 1e-1
    Y_ = Y[torch.randperm(Y.shape[0])].view(Y.shape)
    inner_iterations = 10
    for i in range(ITERS):
        idx = torch.randperm(X.shape[0])[:BATCH_SIZE]
        X_train = X[idx].clone().detach()
        Y_train = Y[idx].clone().detach()
        Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)
        for j in range(inner_iterations):
            T_XY = T.forward(X_train,Y_shuffled)
            f_T = f.forward(T_XY,Y_shuffled) 
            loss_T = - f_T.mean() + 0.5*((X_train-T_XY)*(X_train-T_XY)).sum(axis=1).mean()
            optimizer_T.zero_grad()
            loss_T.backward()
            optimizer_T.step()
                
        f_xy = f.forward(X_train,Y_train) 
        T_XY = T.forward(X_train,Y_shuffled)
        f_T = f.forward(T_XY,Y_shuffled) 
        loss_f = - f_xy.mean() + f_T.mean()
        optimizer_f.zero_grad()
        loss_f.backward()
        optimizer_f.step()

        if  (i+1)==ITERS or i%100==0:
            with torch.no_grad():
                f_xy = f.forward(X,Y) 
                T_XY = T.forward(X,Y_)
                f_T = f.forward(T_XY,Y_) 
                loss_f = f_xy.mean() - f_T.mean()
                loss = f_xy.mean() - f_T.mean() + ((X-T_XY)*(X-T_XY)).sum(axis=1).mean()
                print("Iteration: %d/%d, loss = %.4f" %(i+1,ITERS,loss.item()))
        
        if experiment == "squared" and sigma_w == 0.04: 
            if i>1000:
                scheduler_f.step()
                scheduler_T.step()
        else:
            scheduler_f.step()
            scheduler_T.step()
                    
#%%
N = 1000
# =============================================================================
# NN = [100,500]
# =============================================================================
y_val = 1
L = 1
dy = 1

experiment = "squared"
# experiment = "linear"

sigma_w = 0.4

if experiment == "linear":
# =============================================================================
#     x_lim = 3.3
# =============================================================================
    x_lim = 2.8
    bins = 10
    loc = 2
else:
    x_lim = 3.3 
    bins = 24
    loc = 2



if experiment == "linear":
    sigma = 0.4
    def h(x):
        return x
    
    X = sigma*torch.randn(N,dy) + 2*torch.randint(2,(N,dy))-1
    Y =  X + sigma_w*torch.randn(N,dy)

if experiment == "squared":
    X =  torch.randn(N,L) 
    def h(x):
# =============================================================================
#         return 0.5*x[:,0]*x[:,0]
# =============================================================================
        return 0.5*x*x
    Y = h(X).view(-1,dy) + sigma_w*torch.randn(N,dy)

gamma = sigma_w    

#%%
num_neurons = 32


f  = f_nn(L, dy, num_neurons)
T = T_map(L,dy,num_neurons)
f.apply(init_weights)
T.apply(init_weights)     
#with torch.no_grad():
#    f.layer.weight = torch.nn.parameter.Parameter(torch.nn.functional.relu(f.layer.weight))

ITERS = int(1*1e4)
BATCH_SIZE = int(128)
LR = 1e-3
train(f,T,X,Y,ITERS,LR,BATCH_SIZE)

#%%
plt.figure(figsize = (20, 7.2))
grid = plt.GridSpec(3, 4, wspace =0.2, hspace = 0.2)

Y_shuffled = Y[torch.randperm(Y.shape[0])].view(Y.shape)
x_plot = T.forward(X,Y_shuffled).detach().numpy()
y_plot = Y_shuffled.numpy()

plt.subplot(grid[1:, 0])
# =============================================================================
# plt.figure(figsize=(8,6))
# =============================================================================
plt.plot(X[:,0].numpy(),Y[:,0].numpy(),marker='o',ms=5,ls='none',label=r'$P_{XY}$')
plt.plot(x_plot[:,0],y_plot[:,0],marker='o',ms=5,ls='none',label = r"$S{\#}P_X \otimes P_Y$" , alpha = 0.5)
plt.xlabel(r'$X$')
plt.ylabel(r'$Y$')
plt.axhline(y=y_val,color='k',linestyle = '--')
plt.xlim([-x_lim,x_lim])
if experiment == "squared": 
    if sigma_w == 0.4:
        plt.ylim([-1.9,5])
    else:
        plt.ylim([-0.5,4.5])
plt.legend()
# =============================================================================
# plt.legend(fontsize=16)
# =============================================================================
# =============================================================================
# plt.savefig("%s.pdf" %experiment)
# =============================================================================
# =============================================================================
# plt.show()
# =============================================================================

#%%
y = y_val
xx = np.linspace(-3,3,100)
dx = 6./100

if experiment == "linear":
    px = np.exp(-(xx-1)*(xx-1)/(2*sigma*sigma)) + np.exp(-(xx+1)*(xx+1)/(2*sigma*sigma))
    px = px/np.sum(px*dx)
    pyx =  np.exp(-(y-xx)*(y-xx)/(2*sigma_w*sigma_w))
    pxy = px*pyx
    pxy = pxy/np.sum(pxy*dx)

if experiment == "squared":
    def h_1D(x):
        return (0.5*x*x)
    px = np.exp(-xx*xx/2) 
    px = px/np.sum(px*dx)
    pyx =  np.exp(-(y-h_1D(xx))*(y-h_1D(xx))/(2*sigma_w*sigma_w))
    pxy = px*pyx
    pxy = pxy/np.sum(pxy*dx)   

# =============================================================================
# plt.figure(figsize=(16,6))
# =============================================================================
plt.subplot(grid[0, 0])
# =============================================================================
# plt.subplot(1,2,1)
# =============================================================================
plt.plot(xx,px,label=r"$P_X$")
plt.hist(X[:,0].numpy(),bins=24,density=True,label=r"$X^i$", alpha = 0.5)
plt.legend(loc = 2)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.xlim([-x_lim,x_lim])


if experiment == "squared": 
    bins = np.arange(-x_lim,x_lim,x_lim*2/24)
    if sigma_w == 0.04:
        x_lim = 2.1
        loc = 9
        bins = np.arange(-x_lim,x_lim,x_lim*2/24)


plt.subplot(grid[1, -1])
# =============================================================================
# plt.subplot(1,2,2)
# =============================================================================
plt.plot(xx,pxy,'k')#,label=r"$P_{X|Y=1}$")
X_OT = T.forward(X,y*torch.ones(N,dy))
plt.hist(X_OT.detach().numpy()[:,0],bins=bins,color='r',density=True,label=r"OT", alpha = 0.5)


# =============================================================================
# plt.legend(loc = loc)
# =============================================================================
# =============================================================================
# plt.savefig("%s-p.pdf" %experiment)
# =============================================================================
ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.xlim([-x_lim,x_lim])
#%%
  
y_hat = h(X).view(-1,dy)
eta_EnKF = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)    
y_hatEnKF = y_hat + eta_EnKF

m_hatEnKF = X.mean(axis=0)

#o_hatEnKF = (x_hatEnKF**3).mean(axis=1)
o_hatEnKF = (y_hat).mean(axis=0)

a = (X - m_hatEnKF)

#b = ((x_hatEnKF**3).transpose() - o_hatEnKF)
b = (y_hat - o_hatEnKF)

C_hat_vh = ((a.T@b)/N).view(L,dy)
C_hat_hh = ((b.T@b)/N).view(dy,dy)
K_EnKF = np.matmul(C_hat_vh,np.linalg.inv(C_hat_hh + np.eye(dy)*gamma*gamma))
  
X_EnKF = X + (y*torch.ones(N,dy) - y_hatEnKF) @ K_EnKF.T

plt.subplot(grid[0, -1])
plt.plot(xx,pxy,'k')#,label=r"$P_{X|Y=1}$")
plt.hist(X_EnKF.detach().numpy()[:,0],bins=bins,color='g',density=True,label=r"EnKF", alpha = 0.5)
plt.title('N = {}'.format(N))
# =============================================================================
# plt.hist(X[:,0],bins=24,density=True,label=r"$X_0$", alpha = 0.5)
# =============================================================================

# =============================================================================
# plt.legend(loc = loc)
# =============================================================================

ax = plt.gca()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.xlim([-x_lim,x_lim])




#%% SIR
rng = np.random.default_rng()

X_SIR = X #+ torch.randn(N,L) 
y_hat = h(X_SIR).view(-1,dy)

W = torch.sum((y*torch.ones(N,dy) - y_hat)*(y*torch.ones(N,dy) - y_hat),1)/(2*gamma*gamma)
#print(W)
W = W - torch.min(W)
weight = torch.exp(-W).T
#weight = np.exp(-np.sum((y[i+1,] - h(x_SIR[k,i+1,]).T)*(y[i+1] - h(x_SIR[k,i+1,]).T),axis=1)/(2*gamma*gamma)).T
#W[k,i+1,] = weight/np.sum(weight)
weight = weight/torch.sum(weight)
#x_SIR[k,i+1,0,] = rng.choice(x_SIR[k,i+1,0,], J, p = W[k,i+1,0,])
index = rng.choice(np.arange(N), N, p = weight.detach().numpy())
X_sir = X_SIR[index,:]

plt.subplot(grid[2, -1])
plt.plot(xx,pxy,'k')#,label=r"$P_{X|Y=1}$")
plt.hist(X_sir.detach().numpy()[:,0],bins=bins,density=True,color='b',label=r"SIR", alpha = 0.5)
plt.xlabel('X')
# =============================================================================
# plt.hist(X[:,0],bins=24,density=True,label=r"$X_0$", alpha = 0.5)
# =============================================================================
# =============================================================================
# plt.legend(loc = loc)
# =============================================================================
ax = plt.gca()
# =============================================================================
# ax.get_xaxis().set_visible(False)
# =============================================================================
ax.get_yaxis().set_visible(False)
plt.xlim([-x_lim,x_lim])


# =============================================================================
# sys.exit()
# =============================================================================
#%%
expect_f_ind = 0
expect_T = 0
expect_f_alpha = 0
expect_T_alpha = 0
expect_f_ind_alpha = 0
tot = int(1e5)
epsilion = 0.1
f_index = 0

for i in range(tot):
    N_test = 2
    if experiment == "squared":
        X_test =  torch.randn(N_test,L) 
# =============================================================================
#         Y_test = torch.ones_like(X_test) 
# =============================================================================
        Y_test = h(X_test).view(-1,dy) + sigma_w*torch.randn(N_test,dy)
        
# =============================================================================
#         Y_test = Y_test.detach()
# =============================================================================
        
# =============================================================================
#     Y_shuffled = Y_test[torch.randperm(Y_test.shape[0])].view(Y_test.shape)
# =============================================================================
    if (Y_test[0] > y_val-epsilion and Y_test[0] < y_val+epsilion) and (Y_test[1] > y_val-epsilion and Y_test[1] < y_val+epsilion) :
        Y_test = y_val*torch.ones_like(X_test) 
        jacobian1 = torch.autograd.functional.jacobian(f,(X_test[0].reshape(1,-1),Y_test[0].reshape(1,-1)))[0][0][0][0]
        jacobian2 = torch.autograd.functional.jacobian(f,(X_test[1].reshape(1,-1),Y_test[1].reshape(1,-1)))[0][0][0][0]
        expect_f_ind +=  (torch.dot(X_test[0] - jacobian1 - X_test[1] +jacobian2 , X_test[0] - X_test[1]) < 0)
        if torch.dot(X_test[0] - jacobian1 - X_test[1] +jacobian2 , X_test[0] - X_test[1])<0:
            expect_f_ind_alpha += (torch.dot(X_test[0] - jacobian1 - X_test[1] +jacobian2 , X_test[0] - X_test[1])/ torch.norm(X_test[0] - X_test[1])**2 )
            
        expect_f_alpha += (torch.dot(X_test[0] - jacobian1 - X_test[1] +jacobian2 , X_test[0] - X_test[1])/ torch.norm(X_test[0] - X_test[1])**2 ) 
# =============================================================================
#             f_neg += 1 
# =============================================================================
        f_index += 1.0
        
        
    
    Y_test = y_val*torch.ones_like(X_test)     
    x_test_transport = T.forward(X_test,Y_test).detach()
    expect_T +=  (torch.dot(x_test_transport[0]-x_test_transport[1],X_test[0]-X_test[1]) < 0) 
 
    expect_T_alpha += (torch.dot(x_test_transport[0]-x_test_transport[1],X_test[0]-X_test[1])/ torch.norm(X_test[0] - X_test[1])**2 ) 
    
# =============================================================================
#     expect_f_alpha += (torch.dot(X_test[0] - jacobian1 - X_test[1] +jacobian2 , X_test[0] - X_test[1])/ torch.norm(X_test[0] - X_test[1])**2 ) 
# =============================================================================
    
print(" f=", expect_f_ind/f_index ,expect_f_alpha/f_index, expect_f_ind_alpha/f_index)

print("T=",expect_T/tot ,expect_T_alpha/tot)


# =============================================================================
# print("T=",expect_T/tot ,",  f=", expect_f_ind/f_index,", # of points for f =",expect_T_alpha/tot,expect_f_alpha/f_index, f_index)
# =============================================================================
#%%
plt.figure(figsize=(12,4.8))
grid = plt.GridSpec(1, 2, wspace =0.15, hspace = 0)


xs = torch.linspace(-3, 3, steps=1000)
# =============================================================================
# ys = torch.linspace(-5, 5, steps=100)
# x, y = torch.meshgrid(xs,ys)
# =============================================================================

X_plot = xs.reshape(1000,1)
with torch.no_grad():
    T_plot_transport = T(X_plot,torch.ones_like(X_plot))
    f_plot_transport = (1/2*torch.norm(X_plot,dim=1) - f(X_plot,torch.ones_like(X_plot))[:,0])
    just_f = f(X_plot,torch.ones_like(X_plot))[:,0]


# =============================================================================
# plt.subplot(grid[:, 1:3])
# =============================================================================

plt.subplot(grid[0, 0])

plt.plot(X_plot,f_plot_transport-f_plot_transport.mean(),label=r'$\frac{1}{2} x^2$ - f(X,Y=1)',lw=2)
plt.hist(X_OT.detach().numpy()[:,0],bins=bins,color='b',density=True, alpha = 0.4)
plt.plot(xx,pxy,'k',lw=2, alpha = 0.7,label=r"$P_{X|Y=1}$")
plt.legend(fontsize=14,loc=7)
plt.xlabel('X',fontsize=14)

plt.subplot(grid[0, 1])
plt.plot(X_plot,T_plot_transport,label='T(X,Y=1)',color='r',lw=2)
plt.hist(X.detach().numpy()[:,0],bins=bins,color='r',density=True, alpha = 0.4)
plt.plot(xx,px,'k',lw=2, alpha = 0.7,label=r'$P_X$')
plt.legend(fontsize=14)
plt.xlabel('X',fontsize=14)

