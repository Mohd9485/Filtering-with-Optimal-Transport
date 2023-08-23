#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:55:26 2023

@author: jarrah
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
from torch.distributions.multivariate_normal import MultivariateNormal

plt.rc('font', size=13)          # controls default text sizes

class f_nn(torch.nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim):
        super(f_nn, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.activationSigmoid = nn.Sigmoid()
        self.activationReLu = nn.ReLU()
        self.layer_input = nn.Linear(self.x_dim + self.y_dim, self.hidden_dim, bias=True)
        self.layer_1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layerout = nn.Linear(self.hidden_dim, 1, bias=False)
            
    def forward(self, x,y):
        h = self.layer_input(torch.concat((x,y),dim=1))
        h_temp = self.layer_1(self.activationReLu(h)) 
        z = self.layerout(self.activationReLu(h_temp) + h)  #+ 0.01*(x*x).sum(dim=1)
        return z

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
                
# =============================================================================
#         scheduler_f.step()
#         scheduler_T.step()
# =============================================================================
                    
#%%
N = 10000

experiment = "squared"

# =============================================================================
# experiment = "bimodal"
# =============================================================================


sigma_w = 0.04

if experiment == "bimodal":
    sigma = 0.4
    def h(x):
        return x
    
    X = sigma*torch.randn(N,1) + 2*torch.randint(2,(N,1))-1
    Y =  X + sigma_w*torch.randn(N,1)

if experiment == "squared":
    X =  torch.randn(N,1) 
    def h(x):
        return 0.5*x*x
    Y = h(X) + sigma_w*torch.randn(N,1)

gamma = sigma_w    

plt.plot(X.numpy(),Y.numpy(),marker='o',ms=5,ls='none')
plt.show()

    
#X =  2*torch.rand(N,1) - 1
#Y = 2./3*torch.rand(N,1) - 1./3
#Y[X<-1./3] += (torch.randint(2,Y[X<-1./3].shape)*2-1)*2./3
#Y[X>1./3] += (torch.randint(2,Y[X>1./3].shape)*2-1)*2./3

#%%
num_neurons = 32
f = f_nn(1, 1, num_neurons)
T = T_map(1,1,num_neurons)
f.apply(init_weights)
T.apply(init_weights)     
#with torch.no_grad():
#    f.layer.weight = torch.nn.parameter.Parameter(torch.nn.functional.relu(f.layer.weight))

ITERS = int(1e4)
BATCH_SIZE = 128
LR = 1e-3
train(f,T,X,Y,ITERS,LR,BATCH_SIZE)

#%%
plt.figure(figsize = (20, 7.2))
grid = plt.GridSpec(2, 4, wspace =0.2, hspace = 0.2)

Y_shuffled = Y[torch.randperm(Y.shape[0])].view(Y.shape)
x_plot = T.forward(X,Y_shuffled).detach().numpy()
y_plot = Y_shuffled.numpy()

plt.subplot(grid[0:, 0:2])
# =============================================================================
# plt.figure(figsize=(8,6))
# =============================================================================
plt.plot(X.numpy(),Y.numpy(),marker='o',ms=5,ls='none',label=r'$P_{XY}$')
plt.plot(x_plot,y_plot,marker='o',ms=5,ls='none',label = r"$S{\#}P_X \otimes P_Y$" , alpha = 0.5)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$y$',fontsize=20)
plt.axhline(y=1,color='k',linestyle = '--')
plt.legend(fontsize=16)
# =============================================================================
# plt.savefig("%s.pdf" %experiment)
# =============================================================================
# =============================================================================
# plt.show()
# =============================================================================

#%%
y = 1.0
xx = np.linspace(-3,3,100)
dx = 6./100

if experiment == "bimodal":
    px = np.exp(-(xx-1)*(xx-1)/(2*sigma*sigma)) + np.exp(-(xx+1)*(xx+1)/(2*sigma*sigma))
    px = px/np.sum(px*dx)
    pyx =  np.exp(-(y-xx)*(y-xx)/(2*sigma_w*sigma_w))
    pxy = px*pyx
    pxy = pxy/np.sum(pxy*dx)

if experiment == "squared":
    px = np.exp(-xx*xx/2) 
    px = px/np.sum(px*dx)
    pyx =  np.exp(-(y-h(xx))*(y-h(xx))/(2*sigma_w*sigma_w))
    pxy = px*pyx
    pxy = pxy/np.sum(pxy*dx)   

# =============================================================================
# plt.figure(figsize=(16,6))
# =============================================================================
plt.subplot(grid[0, 2:])
# =============================================================================
# plt.subplot(1,2,1)
# =============================================================================
plt.plot(xx,px,label=r"$P_X$")
plt.hist(X.numpy(),bins=24,density=True,label=r"$X^i$", alpha = 0.5)
plt.legend(fontsize=16,loc = 2)

plt.subplot(grid[1, 2:])
# =============================================================================
# plt.subplot(1,2,2)
# =============================================================================
plt.plot(xx,pxy,label=r"$P_{X|Y=1}$")
plt.hist(T.forward(X,y*torch.ones(N,1)).detach().numpy(),bins=24,density=True,label=r"$T(X^i,Y=1)$", alpha = 0.5)

plt.legend(fontsize=16,loc = 2)
# =============================================================================
# plt.savefig("%s-p.pdf" %experiment)
# =============================================================================
plt.show()










