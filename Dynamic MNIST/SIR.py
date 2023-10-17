import numpy as np
import time
import torch
from torch.autograd import Variable
# =============================================================================
# from scipy.integrate import odeint
# =============================================================================
# =============================================================================
# from scipy.integrate import  RK45
# =============================================================================
def SIR(X,Y,X0,G,A,h,t,tau,Noise,h_index,r):
    X = X.to('cpu')
    Y = Y.to('cpu')
    G = G.to('cpu')
    
    AVG_SIM = X.shape[0]
    N = Y.shape[1]
    L = X.shape[2]
    dy = Y.shape[2]
    J = X0.shape[2]
    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = Noise[4]
    T = N*tau
    
    start_time = time.time()
    x_SIR =  np.zeros((AVG_SIM,N+1,L,J))
    mse_SIR =  np.zeros((N,AVG_SIM))
    #W = np.zeros((AVG_SIM,N,L,J))
    rng = np.random.default_rng()
    for k in range(AVG_SIM):
        x_SIR[k,0,] = X0[k,]
# =============================================================================
#         x = X[k,]
# =============================================================================
        y = np.array(Y[k,])
        
        for i in range(N):
        
# =============================================================================
#             X_sir_torch = torch.from_numpy(x_SIR[k,i+1,]).to(torch.float32)
# =============================================================================
            X_sir_torch = torch.from_numpy(A(x_SIR[k,i,])).to(torch.float32)
            X_sir_torch = X_sir_torch + torch.from_numpy(sigmma*np.random.randn(L,J)).to(torch.float32)
            
            x_SIR[k,i+1,] = X_sir_torch.detach().numpy()
# =============================================================================
#             z = Variable(torch.randn(J, 100))
# =============================================================================
            Z = G(X_sir_torch.T).to('cpu').detach().reshape(-1,28,28)
            
# =============================================================================
#             Z =  A(X_sir_torch.to('cpu').T)
# =============================================================================
            y_hat = np.array((h(Z.detach(),h_index[i])))

# =============================================================================
#             x_SIR[k,i+1,] = x_SIR[k,i,]+ A(x_SIR[k,i,],t[i])*tau + sai_SIR
# =============================================================================
            W = np.sum((y[i,] - y_hat)*(y[i] - y_hat),axis=1)/(2*gamma*gamma)
            #print(W)
            W = W - np.min(W)
            weight = np.exp(-W).T
            #weight = np.exp(-np.sum((y[i+1,] - h(x_SIR[k,i+1,]).T)*(y[i+1] - h(x_SIR[k,i+1,]).T),axis=1)/(2*gamma*gamma)).T
            #W[k,i+1,] = weight/np.sum(weight)
            weight = weight/np.sum(weight)
            #x_SIR[k,i+1,0,] = rng.choice(x_SIR[k,i+1,0,], J, p = W[k,i+1,0,])
            index = rng.choice(np.arange(J), J, p = weight)
            x_SIR[k,i+1,] = x_SIR[k,i+1,:,index].T
        #mse_SIR[:,k] = ((x_SIR[k,].mean(axis=2)-x)*(x_SIR[k,].mean(axis=2)-x)).mean(axis=1)
    #MSE_SIR = mse_SIR.mean(axis=1)
    print("--- SIR time : %s seconds ---" % (time.time() - start_time))
    return x_SIR #,MSE_SIR