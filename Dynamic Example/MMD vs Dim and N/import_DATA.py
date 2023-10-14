import numpy as np
import matplotlib.pyplot as plt
import torch
import time as Time

import matplotlib
import sys

from SIR import SIR

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=13)          # controls default text sizes
#plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
def relu(x):
    a = 0
    return np.maximum(x,a)


def h(x):
    return x*x


def A(x,t=0):
    alpha = 0.1
    L = x.shape[0]
    return (1-alpha)*np.eye(L) @ (x)

def kernel(X,Y,sigma=1):
    return torch.exp(-sigma*torch.cdist(X.T,Y.T)*torch.cdist(X.T,Y.T))

def MMD(XY, XY_target, kernel,sigma = 1/(2*4**2)):
# =============================================================================
#     N = 1000
#     quantile = torch.quantile(torch.cdist(XY[:,:N].T,XY_target.T).reshape(1,-1),q=0.25).item()
#     print(quantile)
# =============================================================================
# =============================================================================
#     sigma = 0.003#1/(2*quantile**2)
# =============================================================================
# =============================================================================
#     sigma = 1/(2*10**2)
# =============================================================================
# =============================================================================
#     print(sigma)
# =============================================================================
    XY = XY.to(device)
    XY_target = XY_target.to(device)
    return torch.sqrt(kernel(XY,XY,sigma=sigma).mean() + kernel(XY_target,XY_target,sigma=sigma).mean() - 2*kernel(XY,XY_target,sigma=sigma).mean())




load_L = np.load('data_file_change_dim.npz',allow_pickle=True) # h(x) = x^2
load_N = np.load('data_file_change_N.npz',allow_pickle=True) # h(x) = x^2

sampled = int(1e4)
true_particles = int(1e6)
device = 'mps'
tau = 1e-1 # timpe step 

#%%
data = {}
for key in load_L:
    print(key)
    data[key] = load_L[key]
    
    
time = data['time']
T = len(time)
Noise = data['Noise']
LL = data['LL']
X_list = data['X_list'].tolist()
Y_list = data['Y_list'].tolist()
X_SIR_list = data['X_SIR_list'].tolist()
X_OT_list = data['X_OT_list'].tolist()
X_EnKF_list = data['X_EnKF_list'].tolist()

# =============================================================================
# time_EnKF =  data['running_time'].tolist()['EnKF']
# time_OT = data['running_time'].tolist()['OT']
# time_SIR = data['running_time'].tolist()['SIR']
# =============================================================================

sigmma0 = Noise[2]
x0_amp = 1/Noise[0]

MMD_EnKF =[]
MMD_OT = []
MMD_SIR = []
sigma =1/(2*4**2)
# =============================================================================
# LL = LL[-2:-1]
# =============================================================================
# =============================================================================
# LL = LL[0:1]
# =============================================================================
for L in LL:
    X_true = X_list[L.astype('str')]
    Y_true = Y_list[L.astype('str')]
    X_EnKF = X_EnKF_list[L.astype('str')]
    X_SIR = X_SIR_list[L.astype('str')]
    X_OT = X_OT_list[L.astype('str')]
    
    
    AVG_SIM = X_OT.shape[0]
    J = X_EnKF.shape[3]
    SAMPLE_SIZE = X_OT.shape[3]
    L = X_true.shape[2]
    
    X0 = np.zeros((AVG_SIM,L,true_particles))
    for k in range(AVG_SIM):    
        X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),true_particles))
    
    X = np.zeros((AVG_SIM,len(time),L,true_particles))
    for i in range(L):
        print(i)
        x = X_true[:,:,i].reshape(AVG_SIM,T,1)
        y = Y_true[:,:,i].reshape(AVG_SIM,T,1)
        x0 = X0[:,i,:].reshape(AVG_SIM,1,true_particles)
        x_true, _ = SIR(x,y,x0,A,h,time,tau,Noise)
        X[:,:,i,:] = x_true.reshape(AVG_SIM,T,true_particles)
        
    mean =  X.mean(axis=3,keepdims=True)
    std = X.std(axis=3,keepdims=True)
    
    X = (X-mean)/std
    X_EnKF = (X_EnKF-mean)/std
    X_OT = (X_OT-mean)/std
    X_SIR = (X_SIR-mean)/std
        
    X = torch.from_numpy(X).to(torch.float32)
    X_EnKF = torch.from_numpy(X_EnKF).to(torch.float32)
    X_SIR = torch.from_numpy(X_SIR).to(torch.float32)
    X_OT = torch.from_numpy(X_OT).to(torch.float32)
    mmd_EnKF = []
    mmd_SIR = []
    mmd_OT = []
    start_time = Time.time()
    for i in range(len(time)):
        print('dim : ',L, ' time :', i)
        result_enkf = 0
        result_sir = 0
        result_ot = 0
        
        for j in range(AVG_SIM):
            x = X[j,i,:,torch.randint(0,X.shape[3],(sampled,))]
# =============================================================================
#             quantile = torch.quantile(torch.cdist(X[j,i,:,:J].T,X_OT[j,i].T).reshape(1,-1),q=0.25).item()
#             sigma = 1/(2*quantile**2)
#             print(quantile)
# =============================================================================
            
            result_enkf += MMD(x, X_EnKF[j,i], kernel,sigma)
            result_sir += MMD(x, X_SIR[j,i], kernel,sigma)
            result_ot += MMD(x, X_OT[j,i], kernel,sigma)
        
        mmd_EnKF.append(result_enkf.item()/AVG_SIM)
        mmd_SIR.append(result_sir.item()/AVG_SIM)
        mmd_OT.append(result_ot.item()/AVG_SIM)

    MMD_EnKF.append(np.mean(mmd_EnKF))
    MMD_SIR.append(np.mean(mmd_SIR))
    MMD_OT.append(np.mean(mmd_OT))
    


plt.figure(figsize=(10,7.2))  
grid = plt.GridSpec(3, 1, wspace =0.15, hspace = 0.1)

g1 = plt.subplot(grid[:, 0])
plt.semilogy(LL,MMD_EnKF,'g--',lw=2,label='EnKF')
plt.semilogy(LL,MMD_OT,'r-.',lw=2,label='OT')
plt.semilogy(LL,MMD_SIR,'b:',lw=2,label='SIR')
plt.xlabel('dim',fontsize=20)
#%%

data = {}
for key in load_N:
    print(key)
    data[key] = load_N[key]
    
    
time = data['time']
Noise = data['Noise']
JJ = data['JJ']
X_list = data['X_list'].tolist()
Y_list = data['Y_list'].tolist()
X_SIR_list = data['X_SIR_list'].tolist()
X_OT_list = data['X_OT_list'].tolist()
X_EnKF_list = data['X_EnKF_list'].tolist()


sigmma0 = Noise[2]
x0_amp = 1/Noise[0]

MMD_EnKF =[]
MMD_OT = []
MMD_SIR = []

sigma = 1/(2*0.6**2)

for J in JJ:
    X_true = X_list[J.astype('str')]
    Y_true = Y_list[J.astype('str')]
    X_EnKF = X_EnKF_list[J.astype('str')]
    X_SIR = X_SIR_list[J.astype('str')]
    X_OT = X_OT_list[J.astype('str')]


    AVG_SIM = X_OT.shape[0]
    J = X_EnKF.shape[3]
    SAMPLE_SIZE = X_OT.shape[3]
    L = X_true.shape[2]
    
    if J==100:
        X0 = np.zeros((AVG_SIM,L,true_particles))
        for k in range(AVG_SIM):    
            X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),true_particles))
        X , _ = SIR(X_true,Y_true,X0,A,h,time,tau,Noise)
        
        mean =  X.mean(axis=3,keepdims=True)
        std = X.std(axis=3,keepdims=True)
        
# =============================================================================
#         mean = X.min(axis=3,keepdims=True)
#         std = X.max(axis=3,keepdims=True)-mean
# =============================================================================
        
        X = (X-mean)/std
        
        X = torch.from_numpy(X).to(torch.float32)   
        
    
      
    X_EnKF = (X_EnKF-mean)/std
    X_OT = (X_OT-mean)/std
    X_SIR = (X_SIR-mean)/std
    
     
    X_EnKF = torch.from_numpy(X_EnKF).to(torch.float32)
    X_SIR = torch.from_numpy(X_SIR).to(torch.float32)
    X_OT = torch.from_numpy(X_OT).to(torch.float32)
    
    mmd_EnKF = []
    mmd_SIR = []
    mmd_OT = []
    start_time = Time.time()
    for i in range(len(time)):
        print('N : ',J, ' time :', i)
        result_enkf = 0
        result_sir = 0
        result_ot = 0
        
        for j in range(AVG_SIM):
            x = X[j,i,:,torch.randint(0,X.shape[3],(sampled,))]
            
# =============================================================================
#             quantile = torch.quantile(torch.cdist(X[j,i].T,X_EnKF[j,i].T).reshape(1,-1),q=0.25).item()
#             print(quantile)
# =============================================================================
            
            result_enkf += MMD(x, X_EnKF[j,i], kernel,sigma)
            result_sir += MMD(x, X_SIR[j,i], kernel,sigma)
            result_ot += MMD(x, X_OT[j,i], kernel,sigma)
        
        mmd_EnKF.append(result_enkf.item()/AVG_SIM)
        mmd_SIR.append(result_sir.item()/AVG_SIM)
        mmd_OT.append(result_ot.item()/AVG_SIM)

    MMD_EnKF.append(np.mean(mmd_EnKF))
    MMD_SIR.append(np.mean(mmd_SIR))
    MMD_OT.append(np.mean(mmd_OT))
    
plt.figure(figsize=(10,7.2))  
grid = plt.GridSpec(3, 1, wspace =0.15, hspace = 0.1)

g1 = plt.subplot(grid[:, 0])
plt.semilogy(JJ,MMD_EnKF,'g--',lw=2,label='EnKF')
plt.semilogy(JJ,MMD_OT,'r-.',lw=2,label='OT')
plt.semilogy(JJ,MMD_SIR,'b:',lw=2,label='SIR')
plt.xlabel('particles',fontsize=20)
plt.xscale('log')


# =============================================================================
# plt.xticks(JJ)
# =============================================================================

g1.set_xticks(JJ)
g1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
