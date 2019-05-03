#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 21:15:59 2019

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
import train


img_dir = './CelebA/img_align_celeba/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Train and Validation dataloader
test_set = CelabALoader.CelebA(data_test)
test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                           batch_size=batch_size, 
                                           shuffle=True)


model = train.CNN()

model.eval()

#Loss function 
#criterion = nn.BCELoss()  
loss_fn = nn.CrossEntropyLoss() 

with torch.no_grad():
    
    correct = 0
    total = 0
    epochLoss = 0
    for i, (images, labels) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader),desc='Testing'):
        #images = images.reshape(-1, 48*48).float().to(device)
        images = images.float().to(device)
        labels = labels.float().to(device)
        
        outputs = model(images)
        print(outputs)
        
         
        
        #Optimizer      
        loss = loss_fn(outputs, labels)
        epochLoss += outputs.shape[0] * loss.item()

        Result = epochLoss/len(test_set)

        print('Test Accuracy of the model on test images is: {} %'.format((Result) * 100))