import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import odeint
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
from EnKF import EnKF
from SIR import SIR
from OT import OT

from scipy.integrate import  RK45
#%matplotlib auto

# =============================================================================
# np.random.seed(101)
# torch.manual_seed(101)
# =============================================================================

# Choose h(x) here, the observation rule
def h(x):
    H = np.zeros((dy,L))  
# =============================================================================
#     for i in range(dy):
#         H[i,i*3] = 1
# =============================================================================
    H[0,0] = 1
    H[1,1] = 1
    H[2,3] = 1
    H[3,4] = 1
    H[4,6] = 1
    H[5,7] = 1
    return H@x #np.matmul(H,x)

def h_torch(x):
    H = torch.zeros((dy,L))
# =============================================================================
#     for i in range(dy):
#         H[i,i*3] = 1    
# =============================================================================
    H[0,0] = 1
    H[1,1] = 1
    H[2,3] = 1
    H[3,4] = 1
    H[4,6] = 1
    H[5,7] = 1
    return H@x #np.matmul(H,x)

def L96(t, x):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
# =============================================================================
#     F = 10 # Force
# =============================================================================
# =============================================================================
#     L = 9 # number of states
# =============================================================================
    d = np.zeros_like(x)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(L):
        d[i] = (x[(i + 1) % L] - x[i - 2]) * x[i - 1] - x[i] + F   
    return d

def ML96(t, x):
    """Lorenz 96 model with constant forcing"""
    # Setting up vector
# =============================================================================
#     F = 10 # Force
# =============================================================================
# =============================================================================
#     L = 9 # number of states
# =============================================================================
    x = x.reshape(L,-1)
    d = np.zeros_like(x)
    # Loops over indices (with operations and Python underflow indexing handling edge cases)
    for i in range(L):
        d[i,:] = (x[(i + 1) % L,:] - x[i - 2,:]) * x[i - 1,:] - x[i,:] + F   
    return d.reshape(-1)

def Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau,rk45):
    
    sai = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),N)
    eta = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)
    
    x = np.zeros((N,L))
    y = np.zeros((N,dy))
    x0 = 25+x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),1)
    x[0,] = x0

# =============================================================================
#     time_RK45 = []
#     time_RK45.append(0)
# =============================================================================
    for i in range(N-1):
        #print(i)
        if rk45:
# =============================================================================
#             x[i+1,] = odeint(L96, x[i,], t[i:i+2])[1,] + sai[i,]   #AMIR: a simple Euler discretization x[i+1] = x[i] + tau*L96(x[i],t[i]) might make this step faster.
# =============================================================================
            solver =  RK45(L96, t[i], x[i,],T,first_step=tau) 
            solver.step()
            x[i+1,] = solver.y + sai[i,]
            
        else:
            x[i+1,:] = x[i,:] + L96(x[i,:],t[i])*tau  + sai[i,:] 
        y[i+1,] = h(x[i+1,]) + eta[i+1,]
    
# =============================================================================
#     # True system
#     for i in range(N-1):
#         x[i+1,] = L63(x[i,],t[i]) + sai[i,]
#         y[i+1,] = h(x[i+1,])+ eta[i+1,]
# =============================================================================
    return x,y#,time_RK45


#%%    
L = 9 # number of states
tau = 0.01 # timpe step 
T = 2 # final time in seconds
F = 10 # Force
N = int(T/tau) # number of time steps T = 20 s
dy = 6 #12 # number of states observed
rk45 = True

noise = np.sqrt(1e0) # noise level std
sigmma = noise # Noise in the hidden state
sigmma0 = noise # Noise in the initial state distribution
gamma = noise # Noise in the observation
x0_amp = noise*10 # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]


J = 1000 # Number of ensembles EnKF
AVG_SIM = 1 # Number of Simulations to average over


# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
parameters['INPUT_DIM'] = [L,dy]
parameters['NUM_NEURON'] =  int(32)
parameters['SAMPLE_SIZE'] = int(J) 
parameters['BATCH_SIZE'] = int(128)
parameters['LearningRate'] = 1e-2
parameters['ITERATION'] = int(1024) 
parameters['Final_Number_ITERATION'] = int(64) #int(64) #ITERATION 
# =============================================================================
# parameters['Time_step'] = N
# =============================================================================

t = np.arange(0.0, tau*N, tau)


SAVE_True_X = np.zeros((AVG_SIM,N,L))
SAVE_True_Y = np.zeros((AVG_SIM,N,dy))
X0 = np.zeros((AVG_SIM,L,J))
for k in range(AVG_SIM):    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma,tau,rk45)
    SAVE_True_X[k,] = x
    SAVE_True_Y[k,] = y
    X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))


       
SAVE_X_EnKF , MSE_EnKF = EnKF(SAVE_True_X,SAVE_True_Y,X0,ML96,h,t,tau,Noise,rk45)
SAVE_X_SIR , MSE_SIR = SIR(SAVE_True_X,SAVE_True_Y,X0,ML96,h,t,tau,Noise,rk45)
SAVE_X_OT , MSE_OT = OT(SAVE_True_X,SAVE_True_Y,X0,parameters,ML96,h_torch,t,tau,Noise,rk45)

#%%
plt.figure()
plt.semilogy(t,MSE_EnKF,'g-.',label="EnKF")
plt.semilogy(t,MSE_OT,'r:',label="OT" )
plt.semilogy(t,MSE_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('log(mse)')
plt.legend()
plt.show()


#%%
plt.figure()   
for l in range(9):
    plt.subplot(3,3,l+1)
    for i in range(J):
        plt.plot(t,SAVE_X_EnKF[0,:,l,i],'C0',alpha = 0.1)
    plt.plot(t,SAVE_True_X[0,:,l],'k--')
    plt.xlabel('time')
    plt.ylabel('EnKF X'+str(l+1))
    plt.show()
 
plt.figure()   
for l in range(9):
    plt.subplot(3,3,l+1)
    for i in range(J):
        plt.plot(t,SAVE_X_SIR[0,:,l,i],'C0',alpha = 0.1)
    plt.plot(t,SAVE_True_X[0,:,l],'k--',label = 'dns')
    plt.xlabel('time')
    plt.ylabel('SIR X'+str(l+1))
    #plt.legend()
    plt.show()

plt.figure()        
for l in range(9):
    #for j in range(AVG_SIM):    
    plt.subplot(3,3,l+1)   
    for i in range(J):
        plt.plot(t,SAVE_X_OT[0,:,l,i],'C0',alpha = 0.1)
    plt.plot(t,SAVE_True_X[0,:,l],'k--')
    plt.xlabel('time')
    plt.ylabel('OT X'+str(l+1))
    plt.show() 


# =============================================================================
# sys.exit()
# =============================================================================
#%%
np.savez('DATA_file.npz',\
    time = t, Y_true = SAVE_True_Y,X_true = SAVE_True_X,\
    X_EnKF = SAVE_X_EnKF , X_OT = SAVE_X_OT , X_SIR = SAVE_X_SIR,\
        MSE_EnKF = MSE_EnKF , MSE_OT=MSE_OT, MSE_SIR = MSE_SIR)
