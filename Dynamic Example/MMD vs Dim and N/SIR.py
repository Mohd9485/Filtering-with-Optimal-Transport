import numpy as np
import time
def SIR(X,Y,X0,A,h,t,tau,Noise):
    AVG_SIM = X.shape[0]
    N = X.shape[1]
    L = X.shape[2]
    dy = Y.shape[2]
    J = X0.shape[2]
    noise = Noise[0]
    sigmma = Noise[1]# Noise in the hidden state
    sigmma0 = Noise[2] # Noise in the initial state distribution
    gamma = Noise[3] # Noise in the observation
    x0_amp = Noise[4]
    
    total_time = time.time()
    x_SIR =  np.zeros((AVG_SIM,N,L,J))
    mse_SIR =  np.zeros((N,AVG_SIM))
    #W = np.zeros((AVG_SIM,N,L,J))
    rng = np.random.default_rng()
    run_time = 0
    for k in range(AVG_SIM):
        print('Sim # '+str(k))
        x_SIR[k,0,] = X0[k,]
        x = X[k,]
        y = Y[k,]
        start_time = time.time()
        for i in range(N-1):
            sai_SIR = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J).transpose()
            x_SIR[k,i+1,] = A(x_SIR[k,i,],t[i]) + sai_SIR
            W = np.sum((y[i+1,] - h(x_SIR[k,i+1,]).T)*(y[i+1] - h(x_SIR[k,i+1,]).T),axis=1)/(2*gamma*gamma)
            #print(W)
            W = W - np.min(W)
            weight = np.exp(-W).T
            #weight = np.exp(-np.sum((y[i+1,] - h(x_SIR[k,i+1,]).T)*(y[i+1] - h(x_SIR[k,i+1,]).T),axis=1)/(2*gamma*gamma)).T
            #W[k,i+1,] = weight/np.sum(weight)
            weight = weight/np.sum(weight)
            #x_SIR[k,i+1,0,] = rng.choice(x_SIR[k,i+1,0,], J, p = W[k,i+1,0,])
            index = rng.choice(np.arange(J), J, p = weight)
            x_SIR[k,i+1,] = x_SIR[k,i+1,:,index].T
            
        run_time += time.time() - start_time
# =============================================================================
#         mse_SIR[:,k] = ((x_SIR[k,].mean(axis=2)-x)*(x_SIR[k,].mean(axis=2)-x)).mean(axis=1)
# =============================================================================
# =============================================================================
#     MSE_SIR = mse_SIR.mean(axis=1)
# =============================================================================
    print("--- SIR time : %s seconds ---" % (time.time() - total_time))
    return x_SIR,run_time/AVG_SIM