"""
This code is intended to do the oscillation example

The 1D system model is
    x(n+1) = A(x(n)) + sai(n) , sai ~ N(0,sigma^2 I)
    y(n+1) = h(x(n+1)) + eta(n+1) , eta ~ N(0,gamma^2 I)

A(x) = (1-alpha)* x    
h(x) = x , |x| , x^2 , x^3

@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
from EnKF import EnKF
from SIR import SIR
# =============================================================================
# from OT_new import OT
# =============================================================================
from OT import OT
from scipy.integrate import odeint
#%matplotlib auto

np.random.seed(101)
torch.manual_seed(101)

# Choose h(x) here, the observation rule
def h(x):
    #return x[0,]
    #return x[:2,]
    return x[::2,]
    #return x[1:,]
# =============================================================================
# # Choose A(x) here, the updates for the hidden state    
# def A(x):
#     dt = 0.1
#     X = np.zeros_like(x)
#     X[0,] = x[0,] + w[0]*dt + A[0]*np.sin(x[1,] - x[0,])*dt   
#     X[1,] = x[1,] + w[1]*dt + A[1]*np.sin(x[0,] - x[1,])*dt
#     return X%(2*np.pi)
# =============================================================================

def L63(x, t):
    """Lorenz 96 model"""
    # Setting up vector
    #L = 3
    d = np.zeros_like(x)
    sigma = 10
    r = 28
    b = 8/3
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    d[0] = sigma*(x[1]-x[0])
    d[1] = x[0]*(r-x[2])-x[1]
    d[2] = x[0]*x[1]-b*x[2]
    return d

def ML63(x, t , particles = 100):
    """Lorenz 63 model"""
    # Setting up vector
    #L = 3
    d = np.zeros_like(x)
    sigma = 10
    r = 28
    b = 8/3
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
# =============================================================================
#     d[0,:] = sigma*(x[1,:]-x[0,:])
#     d[1,:] = x[0,:]*(r-x[2,:])-x[1,:]
#     d[2,:] = x[0,:]*x[1,:]-b*x[2,:]
# =============================================================================
    
    d[0,] = sigma*(x[1,]-x[0,])
    d[1,] = x[0,]*(r-x[2,])-x[1,]
    d[2,] = x[0,]*x[1,]-b*x[2,]
    return d

def Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau):
    sai = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),N)
    eta = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)
    
    x = np.zeros((N,L))
    y = np.zeros((N,dy))
    x0 = 25+x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),1)
    x[0,] = x0

    
    for i in range(N-1):
        #print(i)
        if Odeint:
            x[i+1,] = odeint(L63, x[i,], t[i:i+2])[1,] #+ sai[i,]   
        else:
            x[i+1,:] = x[i,:] + L63(x[i,:],t[i])*tau  #+ sai[i,:] 
        y[i+1,] = h(x[i+1,]) + eta[i+1,]
    
# =============================================================================
#     # True system
#     for i in range(N-1):
#         x[i+1,] = L63(x[i,],t[i]) + sai[i,]
#         y[i+1,] = h(x[i+1,])+ eta[i+1,]
# =============================================================================
    return x,y

# =============================================================================
# #%%
# x = np.random.randn(3,10) 
# y1 = np.zeros((2,10))
# y2 = np.zeros_like(y1)
# x1 = np.zeros_like(x)
# x2 = np.zeros_like(x)
# for i in range(10):
#     x1[:,i] =  L63(x[:,i], 0.01)
#     y1[:,i] = h(x1[:,i])
# x2 = ML63(x, 0.01)
# y2 = h(x2)
# =============================================================================
#%%    
L = 3 # number of states
tau = 1e-2 # timpe step 
T = 5 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s
dy = 2 # number of states observed

noise = np.sqrt(1e1) # noise level std
sigmma = noise/10 # Noise in the hidden state
sigmma0 = noise # Noise in the initial state distribution
gamma = noise # Noise in the observation
x0_amp = noise # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]
Odeint = False

J = 1000 # Number of ensembles EnKF
AVG_SIM = 10 # Number of Simulations to average over

# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32*2)
parameters['SAMPLE_SIZE'] = int(J) 
parameters['BATCH_SIZE'] = int(64)
parameters['LearningRate'] = 1e-1
parameters['ITERATION'] = int(1024) 
parameters['Final_Number_ITERATION'] = int(64) #int(64) #ITERATION 
parameters['Time_step'] = N



t = np.arange(0.0, tau*N, tau)
SAVE_True_X = np.zeros((AVG_SIM,N,L))
SAVE_True_Y = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau)
    SAVE_True_X[k,] = x
    SAVE_True_Y[k,] = y
    X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))

SAVE_X_EnKF , MSE_EnKF = EnKF(SAVE_True_X,SAVE_True_Y,X0,ML63,h,t,tau,Noise,Odeint)
SAVE_X_SIR , MSE_SIR = SIR(SAVE_True_X,SAVE_True_Y,X0,ML63,h,t,tau,Noise,Odeint)
SAVE_X_OT , MSE_OT = OT(SAVE_True_X,SAVE_True_Y,X0,parameters,L63,h,t,tau,Noise,Odeint)

# =============================================================================
# sys.exit()
# =============================================================================
#%%
# =============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(SAVE_True_X[0,:,0],SAVE_True_X[0,:,1],SAVE_True_X[0,:,2],'k--')
# for i in range(J):
#     ax.plot(SAVE_X_EnKF[0,:,0,i],SAVE_X_EnKF[0,:,1,i],SAVE_X_EnKF[0,:,2,i],'C0',alpha = 0.1)
# plt.show()
# sys.exit()
# =============================================================================

# =============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(SAVE_True_X[0,:,0],SAVE_True_X[0,:,1],SAVE_True_X[0,:,2],'k--')
# for i in range(J):
#     ax.plot(SAVE_X_SIR[0,:,0,i],SAVE_X_SIR[0,:,1,i],SAVE_X_SIR[0,:,2,i],'C0',alpha = 0.1)
# plt.show()
# sys.exit()
# =============================================================================
# =============================================================================
# plt.figure()
# for l in range(L):
#      for j in range(AVG_SIM): 
#         plt.subplot(3,1,l+1)
#         plt.plot(t,SAVE_True_X[j,:,l],'k--',label = l)
#         plt.ylabel('X'+str(l+1))
#         plt.legend()
#         plt.show()
# =============================================================================

# =============================================================================
# plt.figure()        
# for l in range(L):
#     for j in range(AVG_SIM):    
#         plt.subplot(3,1,l+1)   
#         for i in range(J):
#             plt.plot(t,SAVE_X_OT[j,:,l,i],alpha = 0.5)
#         plt.plot(t,SAVE_True_X[j,:,l],'k--')
#         plt.xlabel('time')
#         plt.ylabel('OT X'+str(l+1))
#         plt.ylim(-50,50)
#         plt.show()
# =============================================================================

for j in range(1):  
    
    plt.figure()   
    for l in range(L):
        plt.subplot(3,1,l+1)
        for i in range(J):
            plt.plot(t,SAVE_X_EnKF[j,:,l,i],'C0',alpha = 0.1)
        plt.plot(t,SAVE_True_X[j,:,l],'k--')
        plt.xlabel('time')
        plt.ylabel('EnKF X'+str(l+1))
        plt.show()
    
    plt.figure()   
    for l in range(L):
        plt.subplot(3,1,l+1)
        for i in range(J):
            plt.plot(t,SAVE_X_SIR[j,:,l,i],'C0',alpha = 0.1)
        plt.plot(t,SAVE_True_X[j,:,l],'k--',label = 'dns')
        plt.xlabel('time')
        plt.ylabel('SIR X'+str(l+1))
        #plt.legend()
        plt.show()

  
    plt.figure()   
    for l in range(L):
        plt.subplot(3,1,l+1)
        for i in range(J):
            plt.plot(t,SAVE_X_OT[j,:,l,i],'C0',alpha = 0.1)
        plt.plot(t,SAVE_True_X[j,:,l],'k--')
        plt.xlabel('time')
        plt.ylabel('OT X'+str(l+1))
        plt.show()
# =============================================================================
# sys.exit()
# =============================================================================
#%%
# =============================================================================
# plt.figure()
# plt.plot(t,y)
# plt.show()
# =============================================================================
#%%
plt.figure()
plt.plot(t,MSE_EnKF,'g-.',label="EnKF")
plt.plot(t,MSE_OT,'r:',label="OT" )
plt.plot(t,MSE_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('mse')
plt.legend()
plt.show()

plt.figure()
plt.semilogy(t,MSE_EnKF,'g-.',label="EnKF")
plt.semilogy(t,MSE_OT,'r:',label="OT" )
plt.semilogy(t,MSE_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('log(mse)')
plt.legend()
plt.show()

sys.exit()
#%%
# =============================================================================
# for l in range(L):
#     for j in range(AVG_SIM):
#         #j = 35
#         plt.figure()
#         plt.subplot(3,1,1)
#         for i in range(J):
#             plt.plot(t,SAVE_X_EnKF[j,:,l,i],alpha = 0.2)
#         plt.plot(t,SAVE_True_X[j,:,l],'k--',label = 'dns')
#         #plt.xlabel('time')
#         plt.ylabel('EnKF')
#         #plt.title('$X^2_t$')
#         plt.legend()
#         plt.show()
#     
#     
#         plt.subplot(3,1,2)
#     
#         for i in range(J):
#             plt.plot(t,SAVE_X_OT[j,:,l,i],alpha = 0.5)
#         plt.plot(t,SAVE_True_X[j,:,l],'k--',label = 'dns')
#         #plt.xlabel('time')
#         plt.ylabel('OT')
#         plt.legend()
#         plt.show()
#     
#         plt.subplot(3,1,3)
#     
#         for i in range(J):
#             plt.plot(t,SAVE_X_SIR[j,:,l,i],alpha = 0.5)
#         plt.plot(t,SAVE_True_X[j,:,l],'k--',label = 'dns')
#         plt.xlabel('time')
#         plt.ylabel('SIR')
#         plt.legend()
#         plt.show()
# =============================================================================
        
# =============================================================================
# sys.exit()
# =============================================================================
#%%
np.savez('1.11_DATA_file_with_EnKF_10_sim_5sec.npz',\
    time = t, Y_true = SAVE_True_Y,X_true = SAVE_True_X,Noise=Noise,\
    X_EnKF = SAVE_X_EnKF , X_OT = SAVE_X_OT , X_SIR = SAVE_X_SIR,\
        MSE_EnKF = MSE_EnKF , MSE_OT=MSE_OT, MSE_SIR = MSE_SIR)
