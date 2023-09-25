#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:07:11 2023

@author: jarrah
"""

# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from tqdm import tqdm #.notebook
import matplotlib.pyplot as plt
import sys
import time

# =============================================================================
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()       
#         self.fc1 = nn.Linear(100, 256)
#         self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
#         self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
#         self.fc4 = nn.Linear(self.fc3.out_features, 784)
#     
#     # forward method
#     def forward(self, x): 
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         return torch.tanh(self.fc4(x))
#     
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(784, 1024)
#         self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
#         self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
#         self.fc4 = nn.Linear(self.fc3.out_features, 1)
#     
#     # forward method
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.dropout(x, 0.3)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = F.dropout(x, 0.3)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         x = F.dropout(x, 0.3)
#         return torch.sigmoid(self.fc4(x))
# =============================================================================

#%%
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
#%%
def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))
    
    D_output = D(x_real)#.reshape(-1,y_real.shape[1])
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = Variable(torch.randn(bs, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()

def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = Variable(torch.randn(bs, z_dim).to(device))
    y = Variable(torch.ones(bs, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()    

#%%
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
# =============================================================================
# device = "mps"
# =============================================================================
print(device)  # this should print out CUDA
# Device configuration
# =============================================================================
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# =============================================================================

bs = 32

# MNIST Dataset
# =============================================================================
# transform = transforms.Compose([
#     transforms.ToTensor()])#,transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
# =============================================================================

# =============================================================================
# transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# =============================================================================
transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) 
     
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

# build network
z_dim = 100
mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2)

G = Generator().to(device)
D = Discriminator().to(device)

# loss
criterion = nn.BCELoss() 

# optimizer
lr = 0.0002 
G_optimizer = optim.Adam(G.parameters(), lr = lr)#, betas=(0.85,0.9999))
D_optimizer = optim.Adam(D.parameters(), lr = lr)#, betas=(0.85,0.9999))

#%%
n_epoch = 200
start_time = time.time()
# =============================================================================
# for epoch in tqdm(range(1, n_epoch+1)):      \
# =============================================================================
for epoch in range(1, n_epoch+1):       
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D.train()
        G.train()
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
print("--- Training time : %s seconds ---" % (time.time() - start_time))
#%%
with torch.no_grad():
    test_z = Variable(torch.randn(bs, z_dim).to(device))
    generated = G(test_z).to('cpu').detach().reshape(-1,28,28)
    
    cmap = 'gray' #None # 'gray'
    plt.imshow(generated[1,], cmap=cmap)
# =============================================================================
#     save_image(generated.view(generated.size(0), 1, 28, 28), './samples/sample_' + '.png')
# =============================================================================

sys.exit()    
#%%  
torch.save(G.state_dict(), 'GAN_generator_save_new.pt')
torch.save(D.state_dict(), 'GAN_discriminator_save_new.pt')   
    
        