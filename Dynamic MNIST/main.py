import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder, MNIST
from torchvision import transforms
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import sys
from OT import OT
from EnKF import EnKF
import sys
from SIR import SIR

from torch.distributions.multivariate_normal import MultivariateNormal

# =============================================================================
# from tensorboardX import SummaryWriter
# =============================================================================
DEVICE = "cpu"

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
# =============================================================================
# transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# =============================================================================


data = MNIST(root='./data/', train=False, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=len(data) ,shuffle=False)

for x,_ in data_loader:
    Data = x


    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(x.size(0), 784)
# =============================================================================
#         c = self.label_emb(labels)
#         x = torch.cat([x, c], 1)
# =============================================================================
        out = self.model(x)
        return out#.squeeze()
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        z = z.view(z.size(0), 100)
# =============================================================================
#         c = self.label_emb(labels)
#         x = torch.cat([z, c], 1)
# =============================================================================
        out = self.model(z)
        return out#.view(x.size(0), 28, 28)    
        
generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load('GAN_generator.pt'))

generator.to(DEVICE)

discriminator = Discriminator().to(DEVICE)
discriminator.load_state_dict(torch.load('GAN_discriminator.pt'))


#%%
def h(z,a):
    s1 = a[0].item() 
    s2 = a[1].item() 
    e1 = s1 + r
    e2 = s2 + r
    return z[:, s1:e1, s2:e2].reshape(z.shape[0],-1)

def A(x):
    return (1-alpha)*(x**1)
#%%  
L = 100 # number of states
tau = 1 # timpe step 
T = 20 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s
R = [12]#,7,7,7]#,5,5]
# =============================================================================
# r = 6
# =============================================================================

alpha = 0.2

noise = np.sqrt(1e-4) # noise level std
sigmma = (2*alpha - alpha*alpha)**0.5
# =============================================================================
# sigmma = noise*10 # Noise in the hidden state
# =============================================================================
sigmma0 = noise # Noise in the initial state distribution
gamma = noise*10 # Noise in the observation
x0_amp = 1/noise # Amplifiying the initial state 
Noise = [noise,sigmma,sigmma0,gamma,x0_amp]


J = int(1e3) # Number of ensembles EnKF
AVG_SIM = 1 # Number of Simulations to average over
# OT networks parameters
parameters = {}
parameters['normalization'] = 'None' #'MinMax' #'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]

parameters['NUM_NEURON'] =  int(32*10) #64
parameters['SAMPLE_SIZE'] = int(J) 
parameters['BATCH_SIZE'] = int(64) #128
parameters['LearningRate'] = 1e-3
parameters['ITERATION'] = int(1024) 
parameters['Final_Number_ITERATION'] = int(64*16*4) #int(64) #ITERATION 
parameters['Time_step'] = N

t = np.arange(0.0, tau*N, tau) 
examples = 1


save_data_name = ['DATA_file.npz']

dist = MultivariateNormal(torch.zeros(L),sigmma*sigmma*torch.eye(L))
Z = []
for l in range(len(R)):
    r = R[l]
    dy = r*r # number of states observed
    parameters['INPUT_DIM'] = [L,dy]
     
    index = torch.randint(0,L,(examples,1))

    h_index = torch.cat((torch.randint(27-r,28-r,(N,1)),torch.randint(0,28-r,(N,1))),dim=1)
    
    for k in range(AVG_SIM):
        
        X = torch.zeros((AVG_SIM,N+1,L))
        y = torch.zeros((AVG_SIM,N,dy))
        X0 = np.zeros((AVG_SIM,L,J))
        
        
        X[k,0,:] = torch.rand(L)#*2-1
        X = X.to(DEVICE)

        for n in range(N):

            X[k,n+1,:] = A(X[k,n,:]) + sigmma*dist.sample((1,))
            
                  
            with torch.no_grad():
                z_true = generator(X[k,n+1,:].reshape(1,-1)).reshape(-1,28,28)
                y[k,n,] = h(z_true,h_index[n]).reshape(examples,-1) + gamma*torch.randn(1,dy)
                
                
#%%
# =============================================================================
#         X0[k,] = np.random.rand(L,J)*2-1
# =============================================================================
        X0[k,] = np.random.randn(L,J)
        # =============================================================================
        # model = model.to('cpu')
        # =============================================================================
        SAVE_X_EnKF =  EnKF(X,y,X0,generator,A,h,t,tau,Noise,h_index,r)
        SAVE_X_SIR =  SIR(X,y,X0,generator,A,h,t,tau,Noise,h_index,r)
        SAVE_X_OT = OT(X,y,X0,parameters,generator,A,h,t,tau,Noise,h_index,r)

    #%%
    
# =============================================================================
    np.savez(save_data_name[l],time = t, Y_true = y, X_true = X.to('cpu'), X_OT = SAVE_X_OT,
             X_SIR = SAVE_X_SIR, X_EnKF = SAVE_X_EnKF, index_obs = h_index,Noise = Noise)
# =============================================================================


