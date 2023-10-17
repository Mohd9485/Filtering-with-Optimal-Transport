import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import time as Time
from torch import nn
from torch.autograd import Variable
import matplotlib.patches as patches
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=13)          # controls default text sizes

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
      
generator = Generator()
generator.load_state_dict(torch.load('GAN_generator.pt'))

classifer = Classifer_Net()
classifer.load_state_dict(torch.load('classifer.pt'))



load = np.load('DATA_file.npz')


data = {}
for key in load:
    print(key)
    data[key] = load[key]
    
    
time = data['time']
X_True = data['X_true']
Y_True = data['Y_true']

X_OT = data['X_OT']
X_EnKF = data['X_EnKF']
X_SIR = data['X_SIR']

index_obs = data['index_obs']


Noise = data['Noise']

print("Noise level : ", Noise[1])
#%%
AVG_SIM = X_OT.shape[0]
J = X_OT.shape[3]
SAMPLE_SIZE = X_OT.shape[3]
L = X_True.shape[2]
numbers = np.arange(10)
r = int(np.sqrt(Y_True.shape[-1]))
cmap = 'gray' #None # 'gray'

    
#%%
cmap = 'gray' #None # 'gray'
plot_particles = 16

plt.figure(figsize = (20, 12))
grid = plt.GridSpec(plot_particles+3, Y_True.shape[1]+1, wspace =0.05, hspace = 0.0)

plt.subplot(grid[2, 0:])
plt.axhline(y=1,linewidth =2, color='r' )
plt.ylim([0,2])
plt.axis('off')

for j in range(Y_True.shape[1]+1):  
    if j == 0:
        plt.subplot(grid[0, j])
        plt.imshow(np.zeros((28,28)), cmap=cmap)
        plt.axis('off')
        plt.title('t={}'.format(j))#,fontsize=10)
    else:
        plt.subplot(grid[0, j])
        x_plot = torch.tensor(X_True[0,j,:]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        plt.title(j)#,fontsize=10)
    
for j in range(Y_True.shape[1]):
    if j == 0:
        plt.subplot(grid[1, j])
        plt.imshow(np.zeros((r,r)), cmap=cmap)
        plt.axis('off')
# =============================================================================
#         plt.title('t={}'.format(j))
# =============================================================================
    plt.subplot(grid[1, j+1])
    plt.imshow(Y_True[0,j,].reshape(r,r), cmap=cmap)
    plt.axis('off')
# =============================================================================
#     plt.title(j+1,fontsize=10)
# =============================================================================
    
for i in range(1,plot_particles+1):
    for j in range(Y_True.shape[1]+1):
        plt.subplot(grid[2+i, j])
        x_plot = torch.tensor(X_EnKF[0,j,:,i]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
#         z = Variable(torch.randn(1, 100))
#         Z = generator(z,torch.LongTensor(torch.argmax(x_plot,dim=1))).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
plt.suptitle('EnKF',fontsize=20)
 

#################################################################################

plt.figure(figsize = (20, 12))
grid = plt.GridSpec(plot_particles+3, Y_True.shape[1]+1, wspace =0.05, hspace = 0.0)

plt.subplot(grid[2, 0:])
plt.axhline(y=1,linewidth =2, color='r' )
plt.ylim([0,2])
plt.axis('off')

for j in range(Y_True.shape[1]+1):   
    if j == 0:
        plt.subplot(grid[0, j])
        plt.imshow(np.zeros((28,28)), cmap=cmap)
        plt.axis('off')
        plt.title('t={}'.format(j))#,fontsize=10)
    else:
        plt.subplot(grid[0, j])
        x_plot = torch.tensor(X_True[0,j,:]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        plt.title(j)#,fontsize=10)
    
for j in range(Y_True.shape[1]):  
    if j == 0:
        plt.subplot(grid[1, j])
        plt.imshow(np.zeros((r,r)), cmap=cmap)
        plt.axis('off')
# =============================================================================
#         plt.title('t={}'.format(j))
# =============================================================================
    plt.subplot(grid[1, j+1])
    plt.imshow(Y_True[0,j,].reshape(r,r), cmap=cmap)
    plt.axis('off')
# =============================================================================
#     plt.title(j+1,fontsize=10)
# =============================================================================
    
for i in range(1,plot_particles+1):
    for j in range(Y_True.shape[1]+1):
        plt.subplot(grid[2+i, j])
        x_plot = torch.tensor(X_SIR[0,j,:,i]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
#         z = Variable(torch.randn(1, 100))
#         Z = generator(z,torch.LongTensor(torch.argmax(x_plot,dim=1))).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
plt.suptitle('SIR',fontsize=20)

#################################################################################

plt.figure(figsize = (20,12))
# =============================================================================
# plt.figure(figsize = (20, 7.2))
# =============================================================================
grid = plt.GridSpec(plot_particles+3, Y_True.shape[1]+1, wspace =0.05, hspace = 0.0)

plt.subplot(grid[2, 0:])
plt.axhline(y=1,linewidth =2, color='r' )
plt.ylim([0,2])
plt.axis('off')

for j in range(Y_True.shape[1]+1):   
    if j == 0:
        plt.subplot(grid[0, j])
        plt.imshow(np.zeros((28,28)), cmap=cmap)
        plt.axis('off')
        plt.title('t={}'.format(j))#,fontsize=10)
    else:
        plt.subplot(grid[0, j])
        x_plot = torch.tensor(X_True[0,j,:]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        plt.title(j)#,fontsize=10)
    
for j in range(Y_True.shape[1]): 
    if j == 0:
        plt.subplot(grid[1, j])
        plt.imshow(np.zeros((r,r)), cmap=cmap)
        plt.axis('off')
# =============================================================================
#         plt.title('t={}'.format(j))
# =============================================================================
    plt.subplot(grid[1, j+1])
    plt.imshow(Y_True[0,j,].reshape(r,r), cmap=cmap)
    plt.axis('off')
# =============================================================================
#     plt.title(j+1,fontsize=10)
# =============================================================================
    
for i in range(1,plot_particles+1):
    for j in range(Y_True.shape[1]+1):
        plt.subplot(grid[2+i, j])
        x_plot = torch.tensor(X_OT[0,j,:,i]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
#         z = Variable(torch.randn(1, 100))
#         Z = generator(z,torch.LongTensor(torch.argmax(x_plot,dim=1))).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
plt.suptitle('OT',fontsize=20)

#%%
plt.figure(figsize=(20,12))
grid = plt.GridSpec(3, len(time)+1, wspace =0.08, hspace = 0.05)

# =============================================================================
# grid = plt.GridSpec(3, 20, wspace =0.1, hspace = 0.1)
# =============================================================================

bins = np.arange(-0.5,10.5)

# =============================================================================
# for i in range(len(time)+1-10,len(time)+1):
# =============================================================================
# =============================================================================
# for i in range(10,30):
# =============================================================================
true_digits = []
for i in range(len(time)+1):
    with torch.no_grad():
        x = X_True[0,i,:]
        x = torch.tensor(x).to(torch.float32).view(1,100)
        z = generator(x).reshape(1,28,28)
        output = classifer(z).data.max(1, keepdim=False)[1]
        true_digits.append(output.item())

for i in range(len(time)+1):
    with torch.no_grad():
        x = X_EnKF[0,i,:,:]
        x = torch.tensor(x).to(torch.float32)
        x = x.T
        z = generator(x).reshape(-1,1,28,28)
        output = classifer(z).data.max(1, keepdim=False)[1]
    
    plt.subplot(grid[0, i])
    plt.hist(output,bins=bins,density=True,orientation='horizontal',rwidth=0.5,color='g')
    
    plt.axhline(y=true_digits[i],linewidth =2,linestyle='dashed', color='k' )
    
    ax = plt.gca()
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 
    plt.title('{}'.format(i))
    if i == 0:
        plt.yticks(np.arange(0,10))
        ax.get_yaxis().set_visible(True) 
        plt.ylabel('EnKF',fontsize=20)
        #plt.title('t={}'.format(i+1))
        plt.title('t={}'.format(i))

   
    with torch.no_grad():
        x = X_OT[0,i,:,:]
        x = torch.tensor(x).to(torch.float32)
        x = x.T
        z = generator(x).reshape(-1,1,28,28)
        output = classifer(z).data.max(1, keepdim=False)[1]
    
    plt.subplot(grid[1, i])
    plt.hist(output,bins=bins,density=True,orientation='horizontal',rwidth=0.5,color='r')
    
    plt.axhline(y=true_digits[i],linewidth =2,linestyle='dashed', color='k' )
    
    ax = plt.gca()
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 
# =============================================================================
#     plt.title('t={}'.format(i+1))
# =============================================================================
    if i == 0:
        plt.yticks(np.arange(0,10))
        ax.get_yaxis().set_visible(True) 
        plt.ylabel('OT',fontsize=20)
 


       
    with torch.no_grad():
        x = X_SIR[0,i,:,:]
        x = torch.tensor(x).to(torch.float32)
        x = x.T
        z = generator(x).reshape(-1,1,28,28)
        output = classifer(z).data.max(1, keepdim=False)[1]
    
    plt.subplot(grid[2, i])
    plt.hist(output,bins=bins,density=True,orientation='horizontal',rwidth=0.5,color='b')
    
    plt.axhline(y=true_digits[i],linewidth =2,linestyle='dashed', color='k' )
    
    ax = plt.gca()
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 
# =============================================================================
#     plt.title('t={}'.format(i+1))
# =============================================================================
    if i == 0:
        plt.yticks(np.arange(0,10))
        ax.get_yaxis().set_visible(True) 
        plt.ylabel('SIR',fontsize=20)

        
plt.show()    

sys.exit()
#%%   

# =============================================================================
# plt.figure(figsize = (20, 7.2))
# =============================================================================
plt.figure(figsize = (20, 12))
plot_particles = 12
plot_particle = int(plot_particles/3)

grid = plt.GridSpec(plot_particles+5, Y_True.shape[1]+1, wspace =0.05, hspace = 0.0)

plt.subplot(grid[2, 0:])
plt.axhline(y=1,linewidth =2, color='r' )
plt.ylim([0,2])
plt.axis('off')

plt.subplot(grid[3+plot_particle, 0:])
plt.axhline(y=1,linewidth =2, color='r' )
plt.ylim([0,2])
plt.axis('off')

plt.subplot(grid[4+plot_particle*2, 0:])
plt.axhline(y=1,linewidth =2, color='r' )
plt.ylim([0,2])
plt.axis('off')

for j in range(Y_True.shape[1]+1):   
    if j == 0:
        plt.subplot(grid[0, j])
        plt.imshow(np.zeros((28,28)), cmap=cmap)
        plt.axis('off')
        plt.title('t={}'.format(j))#,fontsize=13)
    else:
        plt.subplot(grid[0, j])
        x_plot = torch.tensor(X_True[0,j,:]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')

        plt.title(j)#,fontsize=10)
    
for j in range(Y_True.shape[1]): 
    if j == 0:
        plt.subplot(grid[1, j])
        plt.imshow(np.zeros((r,r)), cmap=cmap)
        plt.axis('off')
# =============================================================================
#         plt.title('t={}'.format(j))
# =============================================================================
    plt.subplot(grid[1, j+1])
    plt.imshow(Y_True[0,j,].reshape(r,r), cmap=cmap)
    plt.axis('off')
# =============================================================================
#     plt.title(j+1,fontsize=10)
# =============================================================================

k=0    
for i in range(3,3+plot_particle):
    for j in range(Y_True.shape[1]+1):
        plt.subplot(grid[i, j])
        x_plot = torch.tensor(X_EnKF[0,j,:,k]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
#         z = Variable(torch.randn(1, 100))
#         Z = generator(z,torch.LongTensor(torch.argmax(x_plot,dim=1))).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        if j==0:
# =============================================================================
#             plt.ylabel('EnKF')
# =============================================================================
            
            ax = plt.gca()
            ax.set_axis_on()
            if i==5:
                ax.set_ylabel('   EnKF',fontsize=20)
            plt.yticks([],fontsize=0)
            plt.xticks([],fontsize=0)
            
        k+=1
        
k=0
for i in range(4+plot_particle,4+plot_particle*2):
    for j in range(Y_True.shape[1]+1):
        plt.subplot(grid[i, j])
        x_plot = torch.tensor(X_OT[0,j,:,k]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
#         z = Variable(torch.randn(1, 100))
#         Z = generator(z,torch.LongTensor(torch.argmax(x_plot,dim=1))).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        
        if j==0:
            ax = plt.gca()
            ax.set_axis_on()
            if i==5*2:
                ax.set_ylabel('   OT',fontsize=20)
            plt.yticks([],fontsize=0)
            plt.xticks([],fontsize=0)
            
        k+=1
        
k=0
for i in range(5+plot_particle*2,5+plot_particle*3):
    for j in range(Y_True.shape[1]+1):
        plt.subplot(grid[i, j])
        x_plot = torch.tensor(X_SIR[0,j,:,k]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
#         z = Variable(torch.randn(1, 100))
#         Z = generator(z,torch.LongTensor(torch.argmax(x_plot,dim=1))).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        
        if j==0:
# =============================================================================
#             plt.ylabel('EnKF')
# =============================================================================
            
            ax = plt.gca()
            ax.set_axis_on()
            if i==5*3:
                ax.set_ylabel('   SIR',fontsize=20)
            plt.yticks([],fontsize=0)
            plt.xticks([],fontsize=0)
            
        k+=1