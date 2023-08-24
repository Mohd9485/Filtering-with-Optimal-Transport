"""
Use this file to import the data and plot all the figure which were used in the 
CDC paper.
    
X(# simulations, # of time steps, # of states, # of samples or particles)
  
@author: Mohammad Al-Jarrah

"""

import numpy as np
import matplotlib.pyplot as plt
import sys

plt.rc('font', size=13)          # controls default text sizes
#plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
def relu(x):
    a = 0
    return np.maximum(x,a)

## This is the data with using normalization in the training and testing
# =============================================================================
# load = np.load('DATA_file_L96_F_10.npz') 
# =============================================================================
load = np.load('1.0_DATA_file_L96_F_10_10_sim_newLoss_2sec_withEnKF.npz') 
# =============================================================================
# load = np.load('1.0_DATA_file_L96_F_10_10_sim_newLoss_2sec_withoutEnKF.npz') 
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

num_plot = int(np.sqrt(L))

# np.savez('1.0_mse_OT_withoutEnKF.npz',mse_OT_without=MSE_OT)
# sys.exit()
mse_OT_without = np.load('1.0_mse_OT_withoutEnKF.npz')['mse_OT_without']

# =============================================================================
# for l in range(L):
#     for j in range(AVG_SIM):
#         plt.figure()
#         plt.subplot(3,1,1)
#         for i in range(J):
#             plt.plot(time,X_EnKF[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--',label = 'True state')
#         #plt.xlabel('time')
#         plt.ylabel('EnKF')
#         plt.title('j = {}'.format(j)) # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
#         plt.legend()
#         plt.show()
#         
#         #plt.figure()
#         plt.subplot(3,1,2)
#         #for j in range(1):
#         for i in range(SAMPLE_SIZE):
#             plt.plot(time,X_OT[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--',alpha=1)
#         #plt.xlabel('time')
#         plt.ylabel('OT')
#         #plt.legend()
#         plt.show()
#     
#         plt.subplot(3,1,3)
#         #for j in range(1):
#         for i in range(SAMPLE_SIZE):
#             plt.plot(time,X_SIR[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--',alpha=1)
#         plt.xlabel('time')
#         plt.ylabel('SIR')
#         #plt.legend()
#         plt.show()
# =============================================================================

#%%
# =============================================================================
# plt.figure()   
# for l in range(L):
#     plt.subplot(num_plot,num_plot,l+1)
#     for i in range(J):
#         plt.plot(time,X_EnKF[0,:,l,i],'C0',alpha = 0.1)
#     plt.plot(time,X_true[0,:,l],'k--',label = 'dns')
#     plt.xlabel('time')
#     #plt.ylabel('OT')
#     #plt.legend()
# =============================================================================
#%%   
# =============================================================================
# plt.figure()   
# for l in range(L):
#     plt.subplot(num_plot,num_plot,l+1)
#     for i in range(J):
#         plt.plot(time,X_OT[0,:,l,i],'C0',alpha = 0.1)
#     plt.plot(time,X_true[0,:,l],'k--',label = 'dns')
#     plt.xlabel('time')
#     #plt.ylabel('OT')
#     #plt.legend()
# =============================================================================
#%%

# =============================================================================
# Performance_mse_Enkf = ((relu(X_EnKF).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) 
# Performance_mse_OT = ((relu(X_OT).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) 
# Performance_mse_SIR = ((relu(X_SIR).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) 
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


# =============================================================================
# plt.figure()
# plt.subplot(1,2,1)
# plt.semilogy(time,MSE_EnKF,'g-.',label="EnKF")
# plt.semilogy(time,MSE_OT,'r:',label="OT" )
# plt.semilogy(time,MSE_SIR,'b:',label="SIR" )
# plt.xlabel('time')
# plt.ylabel('mse')
# plt.title('h(X_t) = $X^2_t$')
# plt.legend()
# plt.show()
# 
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
# =============================================================================
NotObserve = [2,5,8]
k = 0

plt.figure(figsize = (12, 7.2))
# =============================================================================
# grid = plt.GridSpec(2, 4, wspace =0.2, hspace = 0.2)  
# =============================================================================
for l in range(3):
    plt.subplot(3,3,3*l+1)
    for i in range(J):
        plt.plot(time,X_EnKF[k,:,NotObserve[l],i],'C0',alpha = 0.1)
    plt.plot(time,X_true[k,:,NotObserve[l]],'k--')
    plt.xlabel('time')
    plt.ylabel('X'+str(NotObserve[l]+1))
    if l==0:
        plt.title('EnKF')
    # plt.show()
 
# plt.figure()   
for l in range(3):
    plt.subplot(3,3,3*l+2)
    for i in range(J):
        plt.plot(time,X_SIR[k,:,NotObserve[l],i],'C0',alpha = 0.1)
    plt.plot(time,X_true[k,:,NotObserve[l]],'k--',label = 'dns')
    plt.xlabel('time')
    if l==0:
        plt.title('SIR')
    # plt.ylabel('SIR X'+str(NotObserve[l]+1))
    #plt.legend()
    # plt.show()

# plt.figure()        
for l in range(3):
    #for j in range(AVG_SIM):    
    plt.subplot(3,3,3*l+3)   
    for i in range(J):
        plt.plot(time,X_OT[k,:,NotObserve[l],i],'C0',alpha = 0.1)
    plt.plot(time,X_true[k,:,NotObserve[l]],'k--')
    plt.xlabel('time')
    if l==0:
        plt.title('OT')
    # plt.ylabel('OT X'+str(NotObserve[l]+1))
    # plt.show() 
# =============================================================================
plt.show()    
sys.exit()
# =============================================================================
# l = 0
# j=0 # 88
# plt.figure()
# plt.subplot(3,1,1)
# for i in range(J):
#     plt.plot(time,X_EnKF[j,:,l,i],'C0',alpha = 0.1)
# plt.plot(time,X_true[j,:,l],'k--',label = 'True state')
# plt.xlabel('time')
# plt.ylabel('EnKF')
# #plt.title('State$') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
# plt.legend()
# # plt.show()
# 
# #plt.figure()
# plt.subplot(3,1,2)
# #for j in range(1):
# for i in range(SAMPLE_SIZE):
#     plt.plot(time,X_OT[j,:,l,i],'C0',alpha = 0.1)
# plt.plot(time,X_true[j,:,l],'k--',alpha=1)
# plt.xlabel('time')
# plt.ylabel('OT')
# plt.legend()
# # plt.show()
# 
# plt.subplot(3,1,3)
# #for j in range(1):
# for i in range(SAMPLE_SIZE):
#     plt.plot(time,X_SIR[j,:,l,i],'C0',alpha = 0.1)
# plt.plot(time,X_true[j,:,l],'k--',alpha=1)
# plt.xlabel('time')
# plt.ylabel('SIR')
# plt.legend()
# # plt.show()
# 
# #plt.savefig('NonLinearState_XX.pdf') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
# =============================================================================
#%%

plt.figure()
#plt.subplot(1,2,1)
plt.semilogy(time,MSE_EnKF,'g--',label="EnKF")
plt.semilogy(time,MSE_SIR,'b:',label="SIR" )
plt.semilogy(time,MSE_OT,'r-.',label="$OT_{with EnKF}$" )
plt.plot(time,mse_OT_without,'y:',label="$OT_{without EnKF}$" )
plt.xlabel('time')
plt.ylabel('mse')
plt.title('Log-MSE')
plt.legend()
plt.show()
#plt.savefig('NonLinearMSE_XXX.pdf') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%

#sys.exit()

# =============================================================================
# plt.figure()
# #plt.figure()
# # =============================================================================
# # plt.semilogy(time,Performance_mse_Enkf,'g-.',label="EnKF")
# # plt.semilogy(time,Performance_mse_OT,'r:',label="OT" )
# # =============================================================================
# plt.plot(time,Performance_mse_Enkf,'g--',label="EnKF")
# plt.plot(time,Performance_mse_OT,'r-.',label="OT" )
# plt.plot(time,Performance_mse_SIR,'b:',label="SIR" )
# plt.xlabel('time')
# plt.ylabel('mse')
# plt.title('mse=max(0,X)')
# plt.legend()
# plt.show()
# #plt.savefig('NonLinearPerformance_XX.pdf') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
# =============================================================================




