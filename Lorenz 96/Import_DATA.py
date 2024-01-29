import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib


plt.rc('font', size=13)          # controls default text sizes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
def relu(x):
    a = 0
    return np.maximum(x,a)


load = np.load('DATA_file.npz') 
# =============================================================================
# mse_OT_without = np.load('DATA_file_without_EnKF.npz')['MSE_OT']
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

#%%
# =============================================================================
NotObserve = [2,5,8]
k = 0
plt.figure(figsize=(12,7.2))
L=3   
for l in range(L):
    plt.subplot(L,3,l+1)
    for i in range(J):
# =============================================================================
#         plt.plot(time,X_EnKF[k,:,l,i],'C0',alpha = 0.1)
# =============================================================================
        plt.plot(time,X_EnKF[k,:,NotObserve[l],i],'g',alpha = 0.1)
    plt.plot(time,X_true[k,:,NotObserve[l]],'k--',label='True state')
    plt.xlabel('time')
# =============================================================================
#     plt.ylabel('X'+str(l+1))
# =============================================================================
    if l==0:
        plt.ylabel('EnKF',fontsize=20)
        plt.legend()
    plt.title('X(%i)'%(NotObserve[l]+1),fontsize=20)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    
    if l==0:
        plt.ylim([-44,44])
    elif l==1:
        plt.ylim([-64,64]) 
    elif l==2:
        plt.ylim([-34,74]) 

    # plt.show()
 
# plt.figure()        
for l in range(L):
    #for j in range(AVG_SIM):    
    plt.subplot(L,3,l+4)   
    for i in range(J):
# =============================================================================
#         plt.plot(time,X_OT[k,:,l,i],'C0',alpha = 0.1)
# =============================================================================
        plt.plot(time,X_OT[k,:,NotObserve[l],i],'r',alpha = 0.1)
    plt.plot(time,X_true[k,:,NotObserve[l]],'k--')
    plt.xlabel('time')
    # plt.ylabel('OT X'+str(l+1))
    if l==0:
        plt.ylabel('OT',fontsize=20)
    # plt.show() 
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    
    if l==0:
        plt.ylim([-44,44])
    elif l==1:
        plt.ylim([-64,64]) 
    elif l==2:
        plt.ylim([-34,74]) 
        
# plt.figure()   
for l in range(L):
    plt.subplot(L,3,l+7)
    for i in range(J):
# =============================================================================
#         plt.plot(time,X_SIR[k,:,l,i],'C0',alpha = 0.1)
# =============================================================================
        plt.plot(time,X_SIR[k,:,NotObserve[l],i],'b',alpha = 0.1)
    plt.plot(time,X_true[k,:,NotObserve[l]],'k--')
    plt.xlabel('time')
    # plt.ylabel('SIR X'+str(l+1))
    #plt.legend()
    if l==0:
        plt.ylabel('SIR',fontsize=20)
    # plt.show()

    if l==0:
        plt.ylim([-44,44])
    elif l==1:
        plt.ylim([-64,64]) 
    elif l==2:
        plt.ylim([-34,74]) 




plt.figure(figsize=(10,7.2))
#plt.subplot(1,2,1)
plt.semilogy(time,MSE_EnKF,'g--',label="EnKF",lw=2)
plt.semilogy(time,MSE_OT,'r-.',label="$OT_{with EnKF}$" ,lw=2)
# =============================================================================
# plt.plot(time,mse_OT_without,'c-.',label="$OT_{without EnKF}$" ,lw=2)
# =============================================================================
plt.semilogy(time,MSE_SIR,'b:',label="SIR" ,lw=2)
plt.xlabel('time')
plt.ylabel('mse',fontsize=20)
#plt.title('Log-MSE')
plt.legend(fontsize=16)
plt.show()





