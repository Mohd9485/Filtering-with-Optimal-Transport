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
# batch_size = 32
# =============================================================================
# =============================================================================
# batch_size = 100
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
generator.load_state_dict(torch.load('GAN_generator_save_new.pt'))

discriminator = Discriminator().to(DEVICE)
discriminator.load_state_dict(torch.load('GAN_discriminator_save_new.pt'))

# =============================================================================
# z = Variable(torch.randn(100, 100)).to(DEVICE)
# images = generator(z).reshape(-1,1,28,28)
# grid = make_grid(images, nrow=10, normalize=True).to('cpu')
# fig, ax = plt.subplots(figsize=(10,10))
# ax.imshow(grid.permute(1, 2, 0).data, cmap='binary')
# ax.axis('off')
# sys.exit()
# =============================================================================

#%%
# =============================================================================
# test_index = Data[data.targets == 9,]
# z_true = test_index[torch.randint(0, test_index.shape[0], (1,)),].reshape(-1,28,28)
# plt.imshow(z_true[0,])
# sys.exit()
# =============================================================================
#%%
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

# =============================================================================
# [0,15:26,8:20]
# =============================================================================
# =============================================================================
# 17:26,7:20
# =============================================================================
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
parameters['ITERATION'] = int(1)#024*4) 
parameters['Final_Number_ITERATION'] = int(64*16*4) #int(64) #ITERATION 
parameters['Time_step'] = N

t = np.arange(0.0, tau*N, tau) 
examples = 1
blur = True*0


save_data_name = ['DATA_file.npz']


for l in range(len(R)):
    r = R[l]
    dy = r*r # number of states observed
    parameters['INPUT_DIM'] = [L,dy]
    
    X = torch.zeros((AVG_SIM,examples,L))
    y = torch.zeros((AVG_SIM,N,dy))
    X0 = np.zeros((AVG_SIM,L,J))
    
    index = torch.randint(0,L,(examples,1))
# =============================================================================
#     for i in range(examples):
# # =============================================================================
# #         X[0,i,index[i]] = 1
# # =============================================================================
#         X[0,i,ind_fix[l]] = 1 
# =============================================================================
    X = X.to(DEVICE)
    
    #h_index = torch.randint(0,28-r,(N,2))
# =============================================================================
#     if l <= 3:
#         h_index = torch.cat((torch.randint(0,28-r,(N,1)),torch.randint(0,28-r,(N,1))),dim=1)
#     elif l<=7:
#         h_index = torch.cat((torch.randint(14,28-r,(N,1)),torch.randint(0,28-r,(N,1))),dim=1)
#     else:
# =============================================================================
# =============================================================================
#     h_index = torch.cat((torch.randint(20,28-r,(N,1)),torch.randint(0,28-r,(N,1))),dim=1)
# =============================================================================
# =============================================================================
#     h_index = torch.cat((torch.randint(13,13+1,(N,1)),torch.randint(0,28-r,(N,1))),dim=1)
# =============================================================================
# =============================================================================
#     h_index = torch.cat((torch.arange(7,20-r,r)),(torch.arange(8,20-r,r)),dim=1)
# =============================================================================
    
    
    h_index = torch.zeros((len(rows)*len(columns),2))
    for ii in range(len(rows)):
        for jj in range(len(columns)):
            h_index[ii*len(columns)+jj,] = torch.tensor([rows[ii],columns[jj]])

    
    
    for k in range(AVG_SIM):
        generator.to(DEVICE)

# =============================================================================
#         z = generator(X[k,]).to('cpu').detach().reshape(examples,28,28)
# =============================================================================
# =============================================================================
#         z = Variable(torch.randn(examples, 100)).to(DEVICE)
#         z_true = generator(z).to('cpu').detach().reshape(examples,28,28)
#         plt.imshow(z_true[0,])
# =============================================================================
        

        test_index = Data[data.targets == condition_number[l]]
        test_ind = torch.randint(0, test_index.shape[0], (1,))
# =============================================================================
#         test_ind = 116 #37 # 6
# =============================================================================
# =============================================================================
#         test_ind = 29 # 6
# =============================================================================
        # test_ind = 422
        print(test_ind)
# =============================================================================
#         test_ind = 206 # 3
# =============================================================================
# =============================================================================
#         test_ind = 637 # 7
# =============================================================================
        z_true = test_index[test_ind,].reshape(-1,28,28)
       
# =============================================================================
#         z1 = torch.load('condition_number.pt')
#         z2 = torch.load('condition_number2.pt')
#         z_true = (z1+z2)/2
# =============================================================================
# =============================================================================
#         z_true = torch.load('condition_number2.pt')
# =============================================================================
     
        
        blur_image = torch.clone(z_true)

        d = 2
        for i in range(d,28-d,d):
            for j in range(d,28-d,d):
              blur_image[0,i-d:i+d,j-d:j+d] =   torch.mean(z_true[0,i-d:i+d,j-d:j+d]) 
             
              
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(z_true[0,])
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(blur_image[0,])
        plt.axis('off')      
# =============================================================================
#         sys.exit()
# =============================================================================
        
        for j in range(N):
            if blur :
                y[k,j,] = h(blur_image,h_index[j]).reshape(examples,-1) + gamma*torch.randn(1,dy)
            else:
# =============================================================================
#                 y[k,j,] = h(z_true,h_index[j]).reshape(examples,-1) + gamma*torch.randn(1,dy)
# =============================================================================
                y[k,j,] = h_y(z_true,h_index[j]).reshape(examples,-1) + gamma*torch.randn(1,dy)
         
            
# =============================================================================
#         plt.figure()
#         grid = plt.GridSpec(5, 8, wspace =0.05, hspace = 0.05)    
#         kk=0
#         cmap = 'gray' #None # 'gray'
#         for i in range(5):
#             for j in range(8):
#                 plt.subplot(grid[i,j])
#                 y_plot = y[0,kk,:].reshape(r,r)
#                 plt.imshow(y_plot, cmap=cmap)
#         # =============================================================================
#         #         plt.autoscale()
#         # =============================================================================
#                 plt.axis('off')
#                 kk += 1
# =============================================================================
                
                
# =============================================================================
#         X0[k,] =  abs(x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J)))
# =============================================================================
# =============================================================================
#         X0[k,] = np.random.rand(L,J)*2-1
# =============================================================================
        X0[k,] = np.random.randn(L,J)
# =============================================================================
#     sys.exit()
# =============================================================================
    #%%
    # =============================================================================
    # model = model.to('cpu')
    # =============================================================================
    SAVE_X_EnKF =  EnKF(X,y,X0,generator,h,t,tau,Noise,h_index,r)
    SAVE_X_SIR =  SIR(X,y,X0,generator,h,t,tau,Noise,h_index,r)
    SAVE_X_OT = OT(X,y,X0,parameters,generator,h,t,tau,Noise,h_index,r)
    #%%
    # =============================================================================
    # X_OT = np.argmax(SAVE_X_OT, axis=2)
    # x_true = np.argmax(X.to('cpu'), axis=2).item()
    # plt.figure(figsize=(18,12))
    # for i in range(8*8):
    #     plt.subplot(8,8, i+1)
    #     x_plot = torch.tensor(SAVE_X_OT[0,-1,:,i]).to(torch.float32)
    #     x_plot = x_plot.to(DEVICE)
    #     z = model(x_plot.reshape(-1,10))
    #     plt.imshow(z.to('cpu').detach().reshape(28,28))
    #     plt.axis('off')
    # 
    # plt.figure()
    # for i in range(J):
    #     plt.plot(t,X_OT[0,:,i],'C0',alpha=0.1)
    # plt.axhline(y=x_true,color ='k',linestyle='--')
    # 
    # 
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(X_OT[0,0,:],density=True)
    # plt.subplot(2,1,2)
    # plt.hist(X_OT[0,-1,:],density=True)
    # plt.show()
    # 
    # sys.exit()
    # =============================================================================
    
    #%%
    
# =============================================================================
#     np.savez('Data_point', Y_true = y, index_obs = h_index,true_image = z_true)
# =============================================================================
    
# =============================================================================
#     np.savez(save_data_name[l],time = t, Y_true = y, X_true = X.to('cpu'), 
#              true_image = z_true, blur_image= blur_image,test_ind = test_ind.item(),
#              X_OT = SAVE_X_OT, X_SIR = SAVE_X_SIR, X_EnKF = SAVE_X_EnKF, 
#              index_obs = h_index,Noise = Noise)
# =============================================================================

#%%
import torch.nn.functional as F
class Classifer_Net(nn.Module):
    def __init__(self):
        super(Classifer_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)
      

classifer = Classifer_Net()
classifer.load_state_dict(torch.load('classifer.pt'))




plt.figure(figsize=(20,7.2))
grid = plt.GridSpec(3, T+1, wspace =0.1, hspace = 0.1)
# =============================================================================
# grid = plt.GridSpec(3, 10, wspace =0.1, hspace = 0.1)
# =============================================================================

bins = np.arange(-0.5,10.5)

grid_ind =0
for i in range(T+1):
# =============================================================================
# for i in range(T+1-10,T+1):
# =============================================================================
# =============================================================================
# for i in range(10,30):
# =============================================================================
    with torch.no_grad():
        x = SAVE_X_EnKF[0,i,:,:]
        x = torch.tensor(x).to(torch.float32)
        x = x.T
        z = generator(x).reshape(-1,1,28,28)
        output = classifer(z).data.max(1, keepdim=False)[1]
    
    plt.subplot(grid[0, grid_ind])
    plt.hist(output,bins=bins,density=True,orientation='horizontal',rwidth=0.5,color='g')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 
    plt.title('{}'.format(i))
    if grid_ind == 0:
        plt.yticks(np.arange(0,10))
        ax.get_yaxis().set_visible(True) 
        plt.ylabel('EnKF')
        plt.title('t={}'.format(i))

    with torch.no_grad():
        x = SAVE_X_OT[0,i,:,:]
        x = torch.tensor(x).to(torch.float32)
        x = x.T
        z = generator(x).reshape(-1,1,28,28)
        output = classifer(z).data.max(1, keepdim=False)[1]
    
    plt.subplot(grid[1, grid_ind])
    plt.hist(output,bins=bins,density=True,orientation='horizontal',rwidth=0.5,color='r')
    ax = plt.gca()
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 
# =============================================================================
#     plt.title('t={}'.format(i+1))
# =============================================================================
    if grid_ind == 0:
        plt.yticks(np.arange(0,10))
        ax.get_yaxis().set_visible(True) 
        plt.ylabel('OT')    
        
    grid_ind +=1
plt.figure()
plt.subplot(1,2,1)
plt.imshow(z_true[0,])
plt.subplot(1,2,2)
plt.imshow(z_true[0,13:13+r,:])


#%%
cmap = 'gray' #None # 'gray'
h_index = h_index.detach().numpy()
h_index = h_index.astype(int)

def h_inv(y,ind):
    full_x = np.zeros((28,28))
    full_x[ind[0]:ind[0]+r, ind[1]:ind[1]+r] = y.reshape(r,r)
    return full_x

full_y = np.zeros((y.shape[1],28,28))
for j in range(y.shape[1]):
    full_y[j,] = h_inv(y[0,j,], h_index[j,])

avg_imag = np.zeros((y.shape[1],28,28))  
non_zero_ind = full_y != 0  
for k in range(y.shape[1]):    
    Y = full_y[0:k+1,]
    for i in range(28):
        for j in range(28):
            avg = np.mean(Y[non_zero_ind[0:k+1,i,j],i,j])
            if np.isnan(avg):
                avg_imag[k,i,j] = np.min(Y)
            else:    
                avg_imag[k,i,j] = avg
#%%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(avg_imag[-1,], cmap=cmap)
plt.title('Observed average parts')
plt.axis('off')
plt.show()

#%%
# =============================================================================
np.savez('DATA_file.npz',time = t, Y_true = y, X_true = X.to('cpu'), 
         true_image = z_true, blur_image= blur_image,test_ind = test_ind,#.item(),
         X_OT = SAVE_X_OT, X_SIR = SAVE_X_SIR, X_EnKF = SAVE_X_EnKF, 
         index_obs = h_index,Noise = Noise)
# =============================================================================
