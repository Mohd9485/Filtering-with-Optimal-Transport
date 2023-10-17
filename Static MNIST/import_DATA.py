import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import torch
import time as Time
from torch import nn
from torch.autograd import Variable
import matplotlib.patches as patches
from torch.autograd import Variable
import torch.nn.functional as F

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

#################################################################
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

true_image = data['true_image']
index_obs = data['index_obs']


Noise = data['Noise']

test_ind = data['test_ind']

print("Noise level : ", Noise[1])
print("Test ind : ", test_ind)
#%%
AVG_SIM = X_OT.shape[0]
J = X_OT.shape[3]
SAMPLE_SIZE = X_OT.shape[3]
L = X_True.shape[2]
numbers = np.arange(10)
r = int(np.sqrt(Y_True.shape[-1]))

def h_inv(y,ind):
    full_x = np.zeros((28,28))
    full_x[ind[0]:ind[0]+r, ind[1]:ind[1]+r] = y.reshape(r,r)
    return full_x

full_y = np.zeros((Y_True.shape[1],28,28))
for j in range(Y_True.shape[1]):
    full_y[j,] = h_inv(Y_True[0,j,], index_obs[j,])

avg_imag = np.zeros((Y_True.shape[1],28,28))  
non_zero_ind = full_y != 0  
for k in range(Y_True.shape[1]):
    if k==0:
        Y = full_y[0,]
    else:
        Y = full_y[0:k+1,]
    for i in range(28):
        for j in range(28):
            if k==0:
                avg = np.mean(Y[non_zero_ind[0,i,j],i,j])
                if np.isnan(avg):
                    avg_imag[k,i,j] = np.min(Y)
                else:    
                    avg_imag[k,i,j] = avg
            else:
                avg = np.mean(Y[non_zero_ind[0:k+1,i,j],i,j])
                if np.isnan(avg):
                    avg_imag[k,i,j] = np.min(Y)
                else:    
                    avg_imag[k,i,j] = avg

                
#%%
cmap = 'gray' #None # 'gray'
plot_particles = 4
     
plt.figure(figsize = (20, 4.8))
# =============================================================================
# plt.suptitle('EnKF',fontsize=20)
# =============================================================================
grid = plt.GridSpec(plot_particles, plot_particles*4+2, wspace =0.01, hspace = 0.0)
g1 = plt.subplot(grid[0:2, 0:plot_particles-2])
plt.imshow(true_image[0,], cmap=cmap)
plt.axis('off')
plt.title('True image',fontsize=20)

g2 = plt.subplot(grid[2:, 0:plot_particles-2])

# =============================================================================
# plt.imshow(true_image[0,20:], cmap=cmap)
# =============================================================================

plt.imshow(avg_imag[-1,], cmap=cmap)

# =============================================================================
# rect = patches.Rectangle((0, 0), r, r, linewidth=2,
#                          edgecolor='r', facecolor="none")
# =============================================================================
rect = patches.Rectangle((index_obs[0,1],index_obs[0,0]), r, r, linewidth=2,
                         edgecolor='r', facecolor="none")
g2.add_patch(rect)
plt.axis('off')
plt.title('Observed part',fontsize=20)

k=0
for i in range(plot_particles):
    for j in range(plot_particles):
        plt.subplot(grid[0+i, plot_particles+j])
        x_plot = torch.tensor(X_EnKF[0,-1,:,k]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
#         z = Variable(torch.randn(1, 100))
# =============================================================================
# =============================================================================
#         Z = generator(z,torch.LongTensor(torch.argmax(x_plot,dim=1))).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        if k==1:
            plt.title('     EnKF',fontsize=20)
        
        plt.subplot(grid[0+i, 2*plot_particles+j+1])
        x_plot = torch.tensor(X_OT[0,-1,:,k]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
#         z = Variable(torch.randn(1, 100))
#         Z = generator(z,torch.LongTensor(torch.argmax(x_plot,dim=1))).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        if k==1:
            plt.title('     OT',fontsize=20)
            
        plt.subplot(grid[0+i, 3*plot_particles+j+2])
        x_plot = torch.tensor(X_SIR[0,-1,:,k]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        Z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
#         z = Variable(torch.randn(1, 100))
#         Z = generator(z,torch.LongTensor(torch.argmax(x_plot,dim=1))).to('cpu').detach().reshape(-1,28,28)
# =============================================================================
        plt.imshow(Z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        if k==1:
            plt.title('     SIR',fontsize=20)
            
        k += 1

    
#%%
cmap = 'gray' #None # 'gray'
plot_particles = 16
# =============================================================================
# plt.figure(figsize = (20, 7.2))
# =============================================================================
# =============================================================================
# plt.figure(figsize = (8, 12))
# =============================================================================
plt.figure(figsize=(20,12))
grid = plt.GridSpec(plot_particles+2, Y_True.shape[1]+1, wspace =0.0, hspace = 0.0)

# =============================================================================
# for j in range(Y_True.shape[1]):
#     if j==0:
#         plt.subplot(grid[0, j+1])
#         plt.imshow(torch.zeros(28,28), cmap=cmap)
#         plt.axis('off')
#         plt.title(j+1)
#     plt.subplot(grid[0, j+1])
#     plt.imshow(avg_imag[j,], cmap=cmap)
#     plt.axis('off')
#     plt.title(j+1)
# =============================================================================
for j in range(Y_True.shape[1]):  
    if j == 0:
        plt.subplot(grid[0, j])
        plt.imshow(np.zeros((28,28)), cmap=cmap)
        plt.axis('off')
        plt.title('t={}'.format(j))#,fontsize=10)
    plt.subplot(grid[0, j+1])
    plt.imshow(avg_imag[j,], cmap=cmap)
    plt.axis('off')
    plt.title(j+1)#,fontsize=20)

plt.subplot(grid[1, 0:])
plt.axhline(y=1,linewidth =2, color='r' )
plt.ylim([0,2])
plt.axis('off')

for i in range(plot_particles):
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
plt.suptitle('OT', fontsize=20)

# %%
plt.figure(figsize=(20,12))
grid = plt.GridSpec(3, len(time)+1, wspace =0.1, hspace = 0.1)

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

for i in range(len(time)+1):
    with torch.no_grad():
        x = X_EnKF[0,i,:,:]
        x = torch.tensor(x).to(torch.float32)
        x = x.T
        z = generator(x).reshape(-1,1,28,28)
        output = classifer(z).data.max(1, keepdim=False)[1]
    
    plt.subplot(grid[0, i])
    plt.hist(output,bins=bins,density=True,orientation='horizontal',rwidth=0.5,color='g')
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
cmap = 'gray' #None # 'gray'
plot_particles = 16
# =============================================================================
# plt.figure(figsize = (20, 7.2))
# =============================================================================
# =============================================================================
# plt.figure(figsize = (8, 12))
# =============================================================================
plt.figure(figsize=(20,12))
grid = plt.GridSpec(plot_particles+2, Y_True.shape[1]+1, wspace =0.0, hspace = 0.0)

# =============================================================================
# for j in range(Y_True.shape[1]):
#     if j==0:
#         plt.subplot(grid[0, j+1])
#         plt.imshow(torch.zeros(28,28), cmap=cmap)
#         plt.axis('off')
#         plt.title(j+1)
#     plt.subplot(grid[0, j+1])
#     plt.imshow(avg_imag[j,], cmap=cmap)
#     plt.axis('off')
#     plt.title(j+1)
# =============================================================================
for j in range(Y_True.shape[1]):  
    if j == 0:
        plt.subplot(grid[0, j])
        plt.imshow(np.zeros((28,28)), cmap=cmap)
        plt.axis('off')
        plt.title('t={}'.format(j))#,fontsize=10)
    plt.subplot(grid[0, j+1])
    plt.imshow(avg_imag[j,], cmap=cmap)
    plt.axis('off')
    plt.title(j+1)#,fontsize=20)

plt.subplot(grid[1, 0:])
plt.axhline(y=1,linewidth =2, color='r' )
plt.ylim([0,2])
plt.axis('off')

for i in range(plot_particles):
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

plt.suptitle('EnKF', fontsize=20)


cmap = 'gray' #None # 'gray'
plot_particles = 16
# =============================================================================
# plt.figure(figsize = (20, 7.2))
# =============================================================================
# =============================================================================
# plt.figure(figsize = (8, 12))
# =============================================================================
plt.figure(figsize=(20,12))
grid = plt.GridSpec(plot_particles+2, Y_True.shape[1]+1, wspace =0.0, hspace = 0.0)

# =============================================================================
# for j in range(Y_True.shape[1]):
#     if j==0:
#         plt.subplot(grid[0, j+1])
#         plt.imshow(torch.zeros(28,28), cmap=cmap)
#         plt.axis('off')
#         plt.title(j+1)
#     plt.subplot(grid[0, j+1])
#     plt.imshow(avg_imag[j,], cmap=cmap)
#     plt.axis('off')
#     plt.title(j+1)
# =============================================================================
for j in range(Y_True.shape[1]):  
    if j == 0:
        plt.subplot(grid[0, j])
        plt.imshow(np.zeros((28,28)), cmap=cmap)
        plt.axis('off')
        plt.title('t={}'.format(j))#,fontsize=10)
    plt.subplot(grid[0, j+1])
    plt.imshow(avg_imag[j,], cmap=cmap)
    plt.axis('off')
    plt.title(j+1)#,fontsize=20)

plt.subplot(grid[1, 0:])
plt.axhline(y=1,linewidth =2, color='r' )
plt.ylim([0,2])
plt.axis('off')

for i in range(plot_particles):
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
plt.suptitle('SIR', fontsize=20)

plt.show()
#%%
x_plot = torch.tensor(X_SIR[0,-1,:,k]).to(torch.float32)
x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)

ind = 20
# =============================================================================
# grid = plt.GridSpec(ind, ind*2, wspace =0.05, hspace = 0.05)
# =============================================================================

plt.figure(figsize=(12,7.2))
grid = plt.GridSpec(ind+5, ind*2, wspace =0.05, hspace = 0.05)
plt.suptitle('EnKF',fontsize=20)
k=0
for i in range(ind):
    for j in range(ind*2):
        plt.subplot(grid[i, j])
        x_plot = torch.tensor(X_EnKF[0,-1,:,k]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
        plt.imshow(z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        k += 1

    
plt.figure(figsize=(12,7.2))
grid = plt.GridSpec(ind+5, ind*2, wspace =0.05, hspace = 0.05)
plt.suptitle('SIR',fontsize=20)
k=0
for i in range(ind):
    for j in range(ind*2):
        plt.subplot(grid[i, j])
        x_plot = torch.tensor(X_SIR[0,-1,:,k]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
        plt.imshow(z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        k += 1

    
plt.figure(figsize=(12,7.2))
grid = plt.GridSpec(ind+5, ind*2, wspace =0.05, hspace = 0.05)
plt.suptitle('OT',fontsize=20)
k=0
for i in range(ind):
    for j in range(ind*2):
        plt.subplot(grid[i, j])
        x_plot = torch.tensor(X_OT[0,-1,:,k]).to(torch.float32)
        x_plot = x_plot.reshape(1,-1)#.to(DEVICE)
        z = generator(x_plot).to('cpu').detach().reshape(-1,28,28)
        plt.imshow(z.detach().reshape(28,28), cmap=cmap)
        plt.axis('off')
        k += 1  
        
plt.show()
 