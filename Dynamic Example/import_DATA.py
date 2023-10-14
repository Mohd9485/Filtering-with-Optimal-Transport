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

# Choose h(x) here, the observation rule
def h(x):
# =============================================================================
#     return x
# =============================================================================
    return x*x
# =============================================================================
#     return x*x*x
# =============================================================================


def A(x,t=0):
    alpha = 0.1
    L = x.shape[0]
    return (1-alpha)*np.eye(L) @ (x)


load = np.load('DATA_file.npz') # h(x) = x^2



data = {}
for key in load:
    print(key)
    data[key] = load[key]
    
    
time = data['time']
X_true = data['X_true']
Y_true = data['Y_true']
X_EnKF = data['X_EnKF']
X_SIR = data['X_SIR']
X_OT = data['X_OT']
MSE_EnKF = data['MSE_EnKF']
MSE_OT = data['MSE_OT']
MSE_SIR = data['MSE_SIR']
Noise = data['Noise']


AVG_SIM = X_OT.shape[0]
J = X_EnKF.shape[3]
SAMPLE_SIZE = X_OT.shape[3]
L = X_true.shape[2]
tau = 1e-1 # timpe step 
[noise,sigmma,sigmma0,gamma,x0_amp] = Noise

#%%
J_true = J*100
X0 = np.zeros((AVG_SIM,L,J_true))
for k in range(AVG_SIM):    
    X0[k,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J_true))

X , Y = SIR(X_true,Y_true,X0,A,h,time,tau,Noise)

#%%
def kernel(X,Y,method = 'linear', degree=2, offset=0.5, sigma=1, beta=1.5):
    return torch.exp(-sigma*torch.cdist(X.T,Y.T)*torch.cdist(X.T,Y.T))

def MMD(XY, XY_target, kernel):
# =============================================================================
#     N = 1000
# =============================================================================
# =============================================================================
#     quantile = torch.quantile(torch.cdist(XY[:,:N].T,XY_target.T).reshape(1,-1),q=0.25).item()
#     print(quantile)
# =============================================================================
# =============================================================================
#     sigma = 0.003#1/(2*quantile**2)
# =============================================================================
    sigma = 1/(2*1**2)
# =============================================================================
#     print(sigma)
# =============================================================================
    XY = XY.to(device)
    XY_target = XY_target.to(device)
    return torch.sqrt(kernel(XY,XY,sigma=sigma).mean() + kernel(XY_target,XY_target,sigma=sigma).mean() - 2*kernel(XY,XY_target,sigma=sigma).mean())
# =============================================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================================================================




sampled = int(1e4)
device = 'mps'
X = torch.from_numpy(X).to(torch.float32)
X_EnKF = torch.from_numpy(X_EnKF).to(torch.float32)
X_SIR = torch.from_numpy(X_SIR).to(torch.float32)
X_OT = torch.from_numpy(X_OT).to(torch.float32)
mmd_EnKF = []
mmd_SIR = []
mmd_OT = []
start_time = Time.time()
for i in range(len(time)):
    print(i)
    result_enkf = 0
    result_sir = 0
    result_ot = 0
    
    for j in range(AVG_SIM):
        x = X[j,i,:,torch.randint(0,X.shape[3],(sampled,))]
# =============================================================================
#         quantile = torch.quantile(torch.cdist(X[j,i].T,X_EnKF[j,i].T).reshape(1,-1),q=0.25).item()
#         sigma = 1/(2*quantile**2)
# =============================================================================
        result_enkf += MMD(x, X_EnKF[j,i], kernel)
        result_sir += MMD(x, X_SIR[j,i], kernel)
        result_ot += MMD(x, X_OT[j,i], kernel)
    
    mmd_EnKF.append(result_enkf.item()/AVG_SIM)
    mmd_SIR.append(result_sir.item()/AVG_SIM)
    mmd_OT.append(result_ot.item()/AVG_SIM)
print("--- MMD time : %s seconds ---" % (Time.time() - start_time))   

#%%
plt.figure(figsize=(10,7.2))  
grid = plt.GridSpec(3, 1, wspace =0.15, hspace = 0.1)
g1 = plt.subplot(grid[0, 0])

j=0
for i in range(SAMPLE_SIZE):
    plt.plot(time,X_EnKF[j,:,1,i],'g',alpha = 0.1)
plt.plot(time,X_true[j,:,1],'k--',label = 'True state')
#plt.xlabel('time')
plt.ylabel('EnKF',fontsize=20)
plt.legend()

plt.ylim([-6,6])
ax = plt.gca()
ax.get_xaxis().set_visible(False)
# plt.legend()
# plt.show()
       
#plt.figure()
g1 = plt.subplot(grid[1, 0])
#for j in range(1):
for i in range(SAMPLE_SIZE):
    plt.plot(time,X_OT[j,:,1,i],'r',alpha = 0.1)
plt.plot(time,X_true[j,:,1],'k--',alpha=1)
plt.ylim([-6,6])
plt.ylabel('OT',fontsize=20)
ax = plt.gca()
ax.get_xaxis().set_visible(False)
#plt.legend()
# plt.show()

g1 = plt.subplot(grid[2, 0])
#for j in range(1):
for i in range(SAMPLE_SIZE):
    plt.plot(time,X_SIR[j,:,1,i],'b',alpha = 0.1)
plt.plot(time,X_true[j,:,1],'k--',alpha=1)
plt.ylim([-6,6])
plt.xlabel('time',fontsize=20)
plt.ylabel('SIR',fontsize=20)
#plt.legend()


plt.figure(figsize=(10,7.2))  
grid = plt.GridSpec(3, 1, wspace =0.15, hspace = 0.1)

g1 = plt.subplot(grid[:, 0])
plt.plot(time,mmd_EnKF,'g--',label="EnKF",lw=2)
plt.plot(time,mmd_OT,'r-.',label="OT" ,lw=2)
plt.plot(time,mmd_SIR,'b:',label="SIR" ,lw=2)
plt.xlabel('time',fontsize=20)
plt.ylabel('MMD',fontsize=20)
plt.legend(fontsize=16)
plt.show()



