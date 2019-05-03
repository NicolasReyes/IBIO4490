#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 08:39:22 2019

@author: Nico
"""

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
from tqdm import tqdm
import CelebALoader
import time


import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import datasets, transforms, models
from PIL import Image


class CelebA(Dataset):
    
    def __init__(self, img_dir, label_dir, partition_dir, istrain=True, isval=False, transform=None):
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.partition_label = partition_dir
        self.transform = transform
        self.istrain = istrain
        self.isval = isval
        
        images = os.listdir(img_dir)
        images.sort()
        
        with open(partition_dir, 'r') as partition:
            partitions = partition.readlines()
        
        with open(label_dir, 'r') as labels_F:
            labels = labels_F.readlines()
            
        x_train, x_val, x_test, y_train, y_val = [],[],[],[],[]
        #import pdb; pdb.set_trace()
        for i in range(len(images)):
            wild_set = int(partitions[i+1].split(',')[1].split('\n')[0])
            img = images[i]
            if wild_set == 0:
                #import pdb; pdb.set_trace()
                label_string = labels[i+1].split('jpg')[1][1:].split('\n')[0].split(',')
                label = []
                #import pdb; pdb.set_trace()
                for i in range(len(label_string)):
                    for lab in [5,8,9,11,15,17,20,26,31,39]:
                        if label_string[lab] == '-1':
                            label_string[lab] = '0'
                        label.append(int(label_string[lab]))
                x_train.append(img)
                y_train.append(label)
            if wild_set == 1:
                label_string = labels[i+1].split('jpg')[1][1:].split('\n')[0].split(',')
                label = []
                for i in range(len(label_string)):
                    for lab in [5,8,9,11,15,17,20,26,31,39]:
                        if label_string[lab] == '-1':
                            label_string[lab] = '0'
                        label.append(int(label_string[lab]))
                x_val.append(img)
                y_val.append(label)
            if wild_set == 2:
                x_test.append(img)
                
                
        self.train = {'images':x_train, 'labels':y_train}
        self.val = {'images':x_val, 'labels':y_val}
        self.test = {'images':x_test}
        
    def __len__(self):
            
        if self.istrain == True:
            return len(self.train['images'])
        elif self.isval == True:
            return len(self.val['images'])
        else:
            return len(self.val['images'])
        
    def __getitem__(self,idx):
            
        if self.istrain == True:
            img = Image.open(self.img_dir+'/'+self.train['images'][idx])
            img = self.transform(img)
            label = self.train['labels'][idx]
            #import pdb; pdb.set_trace()
            label = torch.LongTensor(label)
        elif self.isval == True:
            img = Image.open(self.img_dir+'/'+self.val['images'][idx])
            img = self.transform(img)
            label = self.train['labels'][idx]
            
            label = torch.LongTensor(label)
        else:
            img = Image.open(self.img_dir+'/'+self.test['images'][idx])
            img = self.transform(img)
            label = []
                
        return img,label
                        
def get_data(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((128,128)),
        transforms.ColorJitter(),
        transforms.ToTensor()
        ])
    img_dir = "/media/user_home2/vision/data/CelebA/img_align_celeba/"
    partition_dir = '/media/user_home2/vision/data/CelebA/train_val_test.txt'
    labels_dir = '/media/user_home2/vision/data/CelebA/list_attr_celeba.txt'

    data_train = CelebA(img_dir = img_dir, label_dir = labels_dir, partition_dir = partition_dir, istrain = True, isval = False, transform = transform_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_val = CelebA(img_dir = img_dir, label_dir = labels_dir, partition_dir = partition_dir , istrain = False, isval = True, transform = transform_train)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True)
    
    data_test = CelebA(img_dir = img_dir, label_dir = labels_dir, partition_dir = partition_dir , istrain = False, isval = False, transform = transform_train)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader



#Initializing hyperparameters
n_epochs = 8
num_classes = 10
batch_size = 10
learning_rate = 0.01

"""
input_size = 48*48
hidden_size = 500
num_classes = 10
n_epochs = 10
batch_size = 75
learning_rate = 0.01
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Train and Validation dataloader
train_loader, val_loader, test_loader = get_data(batch_size)


"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)        #Batch normalization
        self.relu = nn.ReLU()                 #RELU Activation
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)   #Maxpooling reduces the size by kernel size. 64/2 = 32
        
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)    #Size now is 32/2 = 16
        
        #Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is of size 16x16 --> 32*16*16 = 8192
        self.fc1 = nn.Linear(in_features=16384, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
        self.droput = nn.Dropout(p=0.5)                    #Dropout used to reduce overfitting
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50, out_features=2)    #Since there were so many features, I decided to use 45 layers to get output layers. You can increase the kernels in Maxpooling to reduce image further and reduce number of hidden linear layers.
       
        
    def forward(self,x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        #Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1,16384)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        #Then we forward through our fully connected layer 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
        return out
    
"""    

"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)        #Batch normalization
        self.relu = nn.ReLU()                 #RELU Activation
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   #Maxpooling reduces the size by kernel size. 64/2 = 32
        
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)    #Size now is 32/2 = 16
        
        #Flatten the feature maps. You have 32 feature mapsfrom cnn2. Each of the feature is of size 16x16 --> 32*16*16 = 8192
        self.fc1 = nn.Linear(in_features=16384, out_features=4000)   #Flattened image is fed into linear NN and reduced to half size
        self.droput = nn.Dropout(p=0.5)                    #Dropout used to reduce overfitting
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50, out_features=2)    #Since there were so many features, I decided to use 45 layers to get output layers. You can increase the kernels in Maxpooling to reduce image further and reduce number of hidden linear layers.
       
        
    def forward(self,x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        #Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1,16384)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        #Then we forward through our fully connected layer 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
        return out        
"""

class CNN(torch.nn.Module):
    
    #Our batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(CNN, self).__init__()
        
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #16384 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(6*6*128, 500)
        
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(500, num_classes)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 6*6*128)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)
        
"""        
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(22*27*32, 500)
        self.fc2 = nn.Linear(500, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        # print('x',x.shape)
        out = self.layer1(x)
        # print('l1',out.shape)
        out = self.layer2(out)
        # print('l2',out.shape)
        out = self.layer3(out)
        # print('l3',out.shape)
        out = out.reshape(out.size(0), -1)
        # print('reshape',out.shape)
        out = self.relu(self.fc(out))
        # print('final',out.shape)
        out = self.fc2(out)
        # print(out)
        out = self.softmax(out)
        return out

"""

#Define the model
model = CNN()
model.to(device) 

#CUDA = torch.cuda.is_available()
#if CUDA:
#    model = model.cuda()    

#Loss function   
loss_fn = nn.CrossEntropyLoss()  

#Optimizer      
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)





#Define the CNN




#Train the CNN


#Print all of the hyperparameters of the training iteration:
print("===== HYPERPARAMETERS =====")
print("batch_size=", batch_size)
print("epochs=", n_epochs)
print("learning_rate=", learning_rate)
#print("=" * 30)


#Time for printing
training_start_time = time.time()

#Loop for n_epochs
for epoch in range(n_epochs):
    n_batches = batch_size
    running_loss = 0.0
    print_every = n_batches // 10
    start_time = time.time()
    total_train_loss = 0
    
    for i, data in enumerate(train_loader, 0):
        
        #Get inputs
        #inputs, labels = data
        
        #Wrap them in a Variable object
        
        inputs, labels = data
        #import pdb; pdb.set_trace()
        #Set the parameter gradients to zero
        optimizer.zero_grad()
        
        #Forward pass, backward pass, optimize
        outputs = model(inputs.to(device))
        loss_size = loss_fn(outputs, torch.max(labels.to(device),1)[1])
        #import pdb; pdb.set_trace()
        loss_size.backward()
        optimizer.step()
        
        #Print statistics
        running_loss += loss_size.data[0]
        total_train_loss += loss_size.data[0]
        
        #Print every 10th batch of an epoch
        if (i + 1) % (print_every + 1) == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
            #Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
        
    #At the end of the epoch, do a pass on the validation set
    total_val_loss = 0
    for inputs, labels in val_loader:
        
        #Wrap tensors in Variables
        inputs, labels = Variable(inputs), Variable(labels)
        
        #Forward pass
        val_outputs = model(inputs)
        val_loss_size = loss_fn(val_outputs, labels.to(device))
        total_val_loss += val_loss_size.data[0]
        
    print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
    
print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
        
    
    
"""
class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(3,70,kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(70,35,kernel_size=3, stride=1 )
        self.conv3 = nn.Conv2d(35,35,kernel_size=3, stride =1 )
        self.fc1 = nn.Linear(35*4*4,70)
        self.fc2 = nn.Linear(70,7,bias=True)
        
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        #import pdb; pdb.set_trace()
        x = F.avg_pool2d(x, kernel_size = 2, stride = 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        x = F.dropout(x, 0.25, training = self.training)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        x = F.dropout(x, 0.25, training = self.training)
        #import pdb; pdb.set_trace()
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
"""

       