"""
Use this file to import the data and plot all the figure which were used in the 
CDC paper.
    
X(# simulations, # of time steps, # of states, # of samples or particles)
  
@author: Mohammad Al-Jarrah

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import time as Time

# =============================================================================
# from MMD import MMD
# =============================================================================
plt.rc('font', size=13)          # controls default text sizes
#plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
def relu(x):
    a = 0
    return np.maximum(x,a)



load = np.load('0.1_DATA_file_x_newloss_100_sim_withEnKF_101_seed.npz') # h(x) = x^2
X = np.load('0.0_DATA_file_x_trueSIR_100_sim_1e5_particle_101_seed.npz')['X_SIR']


# =============================================================================
# load = np.load('0.1_DATA_file_xx_newloss_1_sim_withEnKF_101_seed.npz') # h(x) = x^2
# X = np.load('0.0_DATA_file_xx_trueSIR_1_sim_1e6_particle_101_seed.npz')['X_SIR']
# =============================================================================


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

#%%
AVG_SIM = X_OT.shape[0]
J = X_EnKF.shape[3]
SAMPLE_SIZE = X_OT.shape[3]
L = X_true.shape[2]


#%%
# =============================================================================
# for j in range(AVG_SIM):   
#     plt.figure()   
#     for l in range(L):
#         plt.subplot(2,1,l+1)
#         for i in range(J):
#             plt.plot(time,X_OT[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--')
#         plt.xlabel('time')
#         plt.ylabel('OT X'+str(l+1))
#         plt.title(j)
#         plt.show()
# sys.exit()
# =============================================================================

#%%  
# =============================================================================
# for j in range(1):  
# # =============================================================================
# # for j in range(AVG_SIM):
# # =============================================================================
#     plt.figure()
#     for l in range(L):
#         plt.subplot(3,2,1+l)
#         for i in range(J):
#             plt.plot(time,X_EnKF[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--',label = 'True state')
#         #plt.xlabel('time')
#         plt.ylabel('EnKF')
#         plt.title('state = {}'.format(l+1)) # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
#         # plt.legend()
#         # plt.show()
#     
#     for l in range(L):       
#         #plt.figure()
#         plt.subplot(3,2,3+l)
#         #for j in range(1):
#         for i in range(SAMPLE_SIZE):
#             plt.plot(time,X_OT[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--',alpha=1)
#         #plt.xlabel('time')
#         plt.ylabel('OT')
#         #plt.legend()
#         # plt.show()
#     
#     for l in range(L):   
#         plt.subplot(3,2,5+l)
#         #for j in range(1):
#         for i in range(SAMPLE_SIZE):
#             plt.plot(time,X_SIR[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--',alpha=1)
#         plt.xlabel('time')
#         plt.ylabel('SIR')
#         #plt.legend()
# plt.show()
# =============================================================================

#%%
for j in range(1):  
# =============================================================================
# for j in range(AVG_SIM):
# =============================================================================
    plt.figure()
    
    plt.subplot(3,1,1)
    for i in range(J):
        plt.plot(time,X_EnKF[j,:,1,i],'C0',alpha = 0.1)
    plt.plot(time,X_true[j,:,1],'k--',label = 'True state')
    #plt.xlabel('time')
    plt.ylabel('EnKF')
    plt.title('$h(X_t)=X_t$') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    # plt.legend()
    # plt.show()
           
    #plt.figure()
    plt.subplot(3,1,2)
    #for j in range(1):
    for i in range(SAMPLE_SIZE):
        plt.plot(time,X_OT[j,:,1,i],'C0',alpha = 0.1)
    plt.plot(time,X_true[j,:,1],'k--',alpha=1)
    #plt.xlabel('time')
    plt.ylabel('OT')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    #plt.legend()
    # plt.show()

    plt.subplot(3,1,3)
    #for j in range(1):
    for i in range(SAMPLE_SIZE):
        plt.plot(time,X_SIR[j,:,1,i],'C0',alpha = 0.1)
    plt.plot(time,X_true[j,:,1],'k--',alpha=1)
    plt.xlabel('time')
    plt.ylabel('SIR')
    #plt.legend()
plt.show()

# =============================================================================
# sys.exit()
# =============================================================================

#%%
def kernel(X,Y,method = 'linear', degree=2, offset=0.5, sigma=1, beta=1.5):
    return torch.exp(-sigma*torch.cdist(X.T,Y.T)*torch.cdist(X.T,Y.T))

def MMD(XY, XY_target, kernel):
    N = 1000
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
plt.figure()
plt.plot(time,mmd_EnKF,'g-.',label="EnKF")
plt.plot(time,mmd_OT,'r:',label="OT" )
plt.plot(time,mmd_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('MMD')
plt.title('$h(X_t)=X_t$')
plt.legend()
plt.show()
sys.exit()
 #%%
# =============================================================================
# Performance_mse_Enkf = ((relu(X_EnKF).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) 
# Performance_mse_OT = ((relu(X_OT).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) 
# Performance_mse_SIR = ((relu(X_SIR).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) 
# =============================================================================



Performance_mse_Enkf = ((relu(X_EnKF).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) + ((abs(X_EnKF).mean(axis = 3) - abs(X_true))**2).mean(axis=(0,2)) 
Performance_mse_OT = ((relu(X_OT).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) +  ((abs(X_OT).mean(axis = 3) - abs(X_true))**2).mean(axis=(0,2))
Performance_mse_SIR = ((relu(X_SIR).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) + ((abs(X_SIR).mean(axis = 3) - abs(X_true))**2).mean(axis=(0,2)) 


# =============================================================================
# np.savez('1.0_xx_Performance_mse_OT_without.npz',Performance_mse_OT_without=Performance_mse_OT)
# sys.exit()
# Performance_mse_OT_without = np.load('1.0_xx_Performance_mse_OT_without.npz')['Performance_mse_OT_without']
# =============================================================================

# =============================================================================
# np.savez('1.0_x_mse_OT_without.npz',mse_OT_without=MSE_OT)
# sys.exit()
# =============================================================================
# mse_OT_without = np.load('1.0_x_mse_OT_without.npz')['mse_OT_without']
# =============================================================================
# =============================================================================

# =============================================================================
# plt.figure()
# plt.plot(time,MSE_EnKF,'g-.',label="EnKF")
# plt.plot(time,MSE_OT,'r:',label="OT" )
# plt.plot(time,MSE_SIR,'b:',label="SIR" )
# plt.xlabel('time')
# plt.ylabel('mse')
# plt.title('MSE')
# plt.legend()
# plt.show()
# =============================================================================


plt.figure()
plt.subplot(1,2,1)
plt.semilogy(time,MSE_EnKF,'g-.',label="EnKF")
plt.semilogy(time,MSE_OT,'r:',label="OT" )
plt.semilogy(time,MSE_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('mse')
plt.title('$h(X_t) = X_t$')
plt.legend()


plt.subplot(1,2,2)
#plt.figure()
# =============================================================================
# plt.semilogy(time,Performance_mse_Enkf,'g-.',label="EnKF")
# plt.semilogy(time,Performance_mse_OT,'r:',label="OT" )
# =============================================================================
plt.semilogy(time,Performance_mse_Enkf,'g-.')
plt.semilogy(time,Performance_mse_OT,'r:')
plt.semilogy(time,Performance_mse_SIR,'b:')
plt.xlabel('time')
plt.ylabel('mse')
plt.title('h(X_t) = $X^2_t$') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
plt.show()

# =============================================================================
# plt.subplot(1,2,2)
# #plt.figure()
# # =============================================================================
# # plt.semilogy(time,Performance_mse_Enkf,'g-.',label="EnKF")
# # plt.semilogy(time,Performance_mse_OT,'r:',label="OT" )
# # =============================================================================
# plt.plot(time,Performance_mse_Enkf,'g-.',label="EnKF")
# plt.plot(time,Performance_mse_OT,'r:')
# plt.plot(time,Performance_mse_SIR,'b:')
# plt.xlabel('time')
# plt.ylabel('mse')
# plt.title('h(X_t) = $X^2_t$') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
# plt.legend()
# plt.show()
# =============================================================================



# =============================================================================
# sys.exit()
# =============================================================================
#%%
l = 1
for j in range(1):  
# for j in range(AVG_SIM):
    # j=0 # 88
    plt.figure()
    plt.subplot(3,1,1)
    for i in range(J):
        plt.plot(time,X_EnKF[j,:,l,i],'C0',alpha = 0.1)
    plt.plot(time,X_true[j,:,l],'k--',label = 'True state')
    plt.xlabel('time')
    plt.ylabel('EnKF')
    #plt.title('State$') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
    plt.legend()


    #plt.figure()
    plt.subplot(3,1,2)
    #for j in range(1):
    for i in range(SAMPLE_SIZE):
        plt.plot(time,X_OT[j,:,l,i],'C0',alpha = 0.1)
    plt.plot(time,X_true[j,:,l],'k--',alpha=1)
    plt.xlabel('time')
    plt.ylabel('OT')
    # plt.legend()


    plt.subplot(3,1,3)
    #for j in range(1):
    for i in range(SAMPLE_SIZE):
        plt.plot(time,X_SIR[j,:,l,i],'C0',alpha = 0.1)
    plt.plot(time,X_true[j,:,l],'k--',alpha=1)
    plt.xlabel('time')
    plt.ylabel('SIR')
    # plt.legend()


#plt.savefig('NonLinearState_XX.pdf') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%


plt.figure()
#plt.subplot(1,2,1)
plt.semilogy(time,MSE_EnKF,'g--',label="EnKF")
plt.semilogy(time,MSE_OT,'r-.',label="$OT_{with EnKF}$")
plt.plot(time,mse_OT_without,'y:',label="$OT_{without EnKF}$" )
plt.semilogy(time,MSE_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('mse')
plt.title('Log-MSE')
plt.legend()
plt.show()
#plt.savefig('NonLinearMSE_XXX.pdf') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%

sys.exit()

# =============================================================================
# print(Performance_mse_OT_without)
# =============================================================================
plt.figure()
#plt.figure()
# =============================================================================
# plt.semilogy(time,Performance_mse_Enkf,'g-.',label="EnKF")
# plt.semilogy(time,Performance_mse_OT,'r:',label="OT" )
# =============================================================================
plt.plot(time,Performance_mse_Enkf,'g--',label="EnKF")
plt.plot(time,Performance_mse_OT,'r-.',label="$OT_{with EnKF}$" )
# =============================================================================
# plt.plot(time,Performance_mse_OT_without,'y:',label="$OT_{without EnKF}$" )
plt.plot(time,mse_OT_without,'y:',label="$OT_{without EnKF}$" )
# =============================================================================
plt.plot(time,Performance_mse_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('mse')
# plt.title('mse=max(0,X)')
plt.legend()
plt.show()
#plt.savefig('NonLinearPerformance_XX.pdf') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%



