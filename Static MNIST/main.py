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

# =============================================================================
# from tensorboardX import SummaryWriter
# =============================================================================
DEVICE = "cpu"

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
# =============================================================================
# transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# =============================================================================

# =============================================================================
# data = MNIST(root='./data/', train=True, download=True, transform=transform)
# =============================================================================
data = MNIST(root='./data/', train=False, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=len(data) ,shuffle=False)

for x,_ in data_loader:
    Data = x


# =============================================================================
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         
#         self.label_emb = nn.Embedding(10, 10)
#         
#         self.model = nn.Sequential(
#             nn.Linear(794, 1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )
#     
#     def forward(self, x, labels):
#         x = x.view(x.size(0), 784)
#         c = self.label_emb(labels)
#         x = torch.cat([x, c], 1)
#         out = self.model(x)
#         return out.squeeze()
#     
# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         
#         self.label_emb = nn.Embedding(10, 10)
#         
#         self.model = nn.Sequential(
#             nn.Linear(110, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 1024),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(1024, 784),
#             nn.Tanh()
#         )
#     
#     def forward(self, z, labels):
#         z = z.view(z.size(0), 100)
#         c = self.label_emb(labels)
#         x = torch.cat([z, c], 1)
#         out = self.model(x)
#         return out.view(x.size(0), 28, 28)
# =============================================================================
    
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

discriminator = Discriminator().to(DEVICE)
discriminator.load_state_dict(torch.load('GAN_discriminator.pt'))

def h_y(z,a):
    s1 = int(a[0].item()) 
    s2 = int(a[1].item())
    e1 = s1 + r
    e2 = s2 + r
    return z[:, s1:e1, s2:e2].reshape(z.shape[0],-1)

def h(z,a):
    s1 = int(a[0].item()) 
    s2 = int(a[1].item())
    e1 = s1 + r
    e2 = s2 + r
    save = z[:, s1:e1, s2:e2]*0
    for i in range(z.shape[0]):
        S1 = s1 + torch.randint(-2,3,(1,)).item()
        S2 = s2 + torch.randint(-2,3,(1,)).item()
        e1 = S1 + r
        e2 = S2 + r
        save[i,] = z[i, S1:e1, S2:e2]
    return save.reshape(z.shape[0],-1)

#%%  

condition_number = [2]
R = [3]#,7,7,7]#,5,5]
r = R[0]

L = 100 # number of states
tau = 1 # timpe step 


rows = torch.arange(17,27-r,r)
columns = torch.arange(7,24-r,r)
T = len(rows)*len(columns) # final time in seconds
N = int(T/tau) # number of time steps T = 20 s
# =============================================================================
# r = 6
# =============================================================================


noise = np.sqrt(1e-4) # noise level std
sigmma = noise*10 # Noise in the hidden state
sigmma0 = noise # Noise in the initial state distribution
gamma = noise*10  # Noise in the observation
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
parameters['ITERATION'] = int(1024*4) 
parameters['Final_Number_ITERATION'] = int(64*16*4) #int(64) #ITERATION 
parameters['Time_step'] = N

t = np.arange(0.0, tau*N, tau) 
examples = 1


save_data_name = ['DATA_file.npz']


for l in range(len(R)):
    r = R[l]
    dy = r*r # number of states observed
    parameters['INPUT_DIM'] = [L,dy]
    
    X = torch.zeros((AVG_SIM,examples,L))
    y = torch.zeros((AVG_SIM,N,dy))
    X0 = np.zeros((AVG_SIM,L,J))
    
    index = torch.randint(0,L,(examples,1))

    X = X.to(DEVICE)
    
    
    h_index = torch.zeros((len(rows)*len(columns),2))
    for ii in range(len(rows)):
        for jj in range(len(columns)):
            h_index[ii*len(columns)+jj,] = torch.tensor([rows[ii],columns[jj]])

    
    for k in range(AVG_SIM):
        generator.to(DEVICE)
        

        test_index = Data[data.targets == condition_number[l]]
        test_ind = torch.randint(0, test_index.shape[0], (1,))

        test_ind = 422
        
        print(test_ind)

        z_true = test_index[test_ind,].reshape(-1,28,28)
    
        for j in range(N):
            y[k,j,] = h_y(z_true,h_index[j]).reshape(examples,-1) + gamma*torch.randn(1,dy)
         
    #%%
    # =============================================================================
    # model = model.to('cpu')
    # =============================================================================
    SAVE_X_EnKF =  EnKF(X,y,X0,generator,h,t,tau,Noise,h_index,r)
    SAVE_X_SIR =  SIR(X,y,X0,generator,h,t,tau,Noise,h_index,r)
    SAVE_X_OT = OT(X,y,X0,parameters,generator,h,t,tau,Noise,h_index,r)

h_index = h_index.detach().numpy()
h_index = h_index.astype(int)
#%%
# =============================================================================
np.savez('DATA_file.npz',time = t, Y_true = y, X_true = X.to('cpu'), 
         true_image = z_true,test_ind = test_ind,#.item(),
         X_OT = SAVE_X_OT, X_SIR = SAVE_X_SIR, X_EnKF = SAVE_X_EnKF, 
         index_obs = h_index,Noise = Noise)
# =============================================================================
