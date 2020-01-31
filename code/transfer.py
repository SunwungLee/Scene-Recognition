#!/usr/bin/env python
# coding: utf-8
"""
@authors: yg6m19 / sz1f19 / sl7a19 (Team: TFA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from os.path import exists, isdir, basename, join, splitext
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image

torch.__version__
from glob import glob
import numpy as np
import os

# In[]: definitions of functions

def get_parameters(path):
    means = [0,0,0]
    std = [0,0,0]
    folders = [files for files in glob(path + "/*") if isdir(files)]
    categories = [basename(folders) for folders in folders]
    train_set = []
    for name in categories:
        ls = os.listdir(path + "/" + name)
        #ls.sort(key= lambda x:int(x[:-4]))
        for filename in ls:
            if filename != '.DS_Store':
                image_path = os.path.join(path + "/" + name,filename)
                train_set += [image_path]
    train_set = np.array(train_set)
    for i in range(len(train_set)):
        img=  Image.open(train_set[i]).convert('RGB')
        img = np.asarray(img)/255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            std[i] += img[:, :, i].std()
    means = np.asarray(means) / len(train_set)
    std = np.asarray(std) / len(train_set)

    return means,std

def train(model,device, train_loader, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x,y= data
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        y_hat= model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch,loss.item()))

# this code was programmed based on ImageNet data
# that is why we don't use it now
#
#def test(model, device, valid_loader):
#    model.eval()
#    test_loss = 0
#    correct = 0
#    with torch.no_grad():
#        for i,data in enumerate(valid_loader):         
#            x,y= data
#            x=x.to(device)
#            y=y.to(device)
#            optimizer.zero_grad()
#            y_hat = model(x)
#            test_loss += criterion(y_hat, y).item() # sum up batch loss
#            pred = y_hat.max(1, keepdim=True)[1] # get the index of the max log-probability
#            correct += pred.eq(y.view_as(pred)).sum().item()
#    test_loss /= len(valid_loader.dataset)
#    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
#        test_loss, correct, len(valid_dataset),
#        100. * correct / len(valid_dataset)))
   
# In[]: data path and set variables

BATCH_SIZE=25
EPOCHS=20
DEVICE = torch.device("cuda:0")     # using CUDA
ngpu= 1
torch.cuda.set_device(0)

train_path = './training/training/'
#valid_path = './test/'
test_path = './testing/testing/'

means,std = get_parameters(train_path) # get parameters based on training data
# means = [0.456]
# std = [0.225]

size = [224,224]
transform  = transforms.Compose(
        [transforms.Resize(size),
         transforms.ToTensor(),
         transforms.Normalize(means, std)])

# In[]: load ResNet152 model

train_dataset = ImageFolder(train_path,transform)
#valid_dataset = ImageFolder(valid_path,transform)
Train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
#Valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=True)

model = models.resnet152(pretrained = True) # Download the pre-training model

print(model.conv1)
num_fc_ftr = model.fc.in_features
for param in model.parameters():
    param.requires_grad = False # Don't change parameters in pre-training model
   
model.fc = nn.Linear(num_fc_ftr, 15) # Define a new fully connected layer
model=model.to(DEVICE)

# In[]: set the loss and optimizer + transfer learning

criterion = nn.CrossEntropyLoss()               # loss function
optimizer = torch.optim.Adam([                  # Adam optimisation algorithm 
    {'params':model.fc.parameters()}
], lr=0.0005)
for epoch in range(1, EPOCHS+1):
    train(model, DEVICE, Train_loader, epoch)   # transfer learning
#    test(model, DEVICE, Valid_loader)           # test using ImageNet data

# In[]:  make run3.txt file

test_set = []                                   # test data given by corsework
files_test = os.listdir(test_path)
output=''

files_test.sort(key= lambda x:int(x[:-4]))
model.eval()
for filename in files_test:
    image_path = os.path.join(test_path,filename)
    test_set += [image_path];
   
test_set = np.array(test_set)

for i in range(len(test_set)):
    img=  Image.open(test_set[i]).convert('RGB')
    img = transform(img)
    img = img.to(DEVICE)
    img = img.unsqueeze(0)
    pred = model(img).max(1, keepdim=True)[1]
    output = output + files_test[i]+' ' + train_dataset.classes[pred] +'\n'

path_file_name = './run3.txt'
str_data = "1\n2"
if not os.path.exists(path_file_name):
    with open(path_file_name, "w") as f:
        print(f)
    with open(path_file_name, "a") as f:
        f.write(output)