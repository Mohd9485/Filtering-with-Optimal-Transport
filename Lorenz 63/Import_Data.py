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
Noise = data['Noise']

#%%
AVG_SIM = X_OT.shape[0]
J = X_EnKF.shape[3]
SAMPLE_SIZE = X_OT.shape[3]
L = X_true.shape[2]

#%%
k = 0
plt.figure(figsize=(12,7.2))   
for l in range(L):
    plt.subplot(L,3,l+1)
    for i in range(J):
# =============================================================================
#         plt.plot(time,X_EnKF[k,:,l,i],'C0',alpha = 0.1)
# =============================================================================
        plt.plot(time,X_EnKF[k,:,l,i],'g',alpha = 0.1)
    plt.plot(time,X_true[k,:,l],'k--',label='True state')
    plt.xlabel('time')
# =============================================================================
#     plt.ylabel('X'+str(l+1))
# =============================================================================
    if l==0:
        plt.ylabel('EnKF')
        plt.legend()
    plt.title('X(%i)'%(l+1))
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
        plt.plot(time,X_OT[k,:,l,i],'r',alpha = 0.1)
    plt.plot(time,X_true[k,:,l],'k--')
    plt.xlabel('time')
    # plt.ylabel('OT X'+str(l+1))
    if l==0:
        plt.ylabel('OT')
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
        plt.plot(time,X_SIR[k,:,l,i],'b',alpha = 0.1)
    plt.plot(time,X_true[k,:,l],'k--')
    plt.xlabel('time')
    # plt.ylabel('SIR X'+str(l+1))
    #plt.legend()
    if l==0:
        plt.ylabel('SIR')
    # plt.show()

    if l==0:
        plt.ylim([-44,44])
    elif l==1:
        plt.ylim([-64,64]) 
    elif l==2:
        plt.ylim([-34,74]) 
    


#%%
plt.figure()
#plt.subplot(1,2,1)
plt.semilogy(time,MSE_EnKF,'g--',label="EnKF")
plt.semilogy(time,MSE_OT,'r-.',label="$OT_{with EnKF}$" )
# =============================================================================
# plt.plot(time,mse_OT_without,'c-.',label="$OT_{without EnKF}$" )
# =============================================================================
plt.semilogy(time,MSE_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('mse')
#plt.title('Log-MSE')
plt.legend()
plt.show()





