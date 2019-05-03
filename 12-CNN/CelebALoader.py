# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 18:01:48 2019

@author: Juan David Triana
"""

import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import datasets, transforms, models
from PIL import Image


class CelebA(Dataset):
    
    def __init__(self, img_dir, labels_dir, partition_dir, istrain=True, isval=False, transform=None):
        
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.partition_label = partition_dir
        self.transform = transform
        self.train = istrain
        self.val = isval
        
        images = os.listdir(img_dir)
        images = images.sort()
        
        with open(partition_dir, 'r') as partition:
            partitions = partition.readlines()
        
        with open(labels_dir, 'r') as labels_F:
            labels = labels_F.readlines()
            
        x_train, x_val, x_test, y_train, y_val, y_test = [],[],[],[],[], []
        
        for i in range(len(images)):
            wild_set = int(partitions[i+1].split(',')[1].split('\n')[0])
            img = images[i]
            if wild_set == 0:
                label_string = labels[i+1].split('jpg'[1][1:].split('\n')[0].split(','))
                label = []
                for i in range(label_string):
                    for lab in [5,8,9,11,15,17,20,26,31,39]:
                        label.append(int(label_string[lab]))
                x_train.append(img)
                y_train.append(label)
            if wild_set == 1:
                label_string = labels[i+1].split('jpg'[1][1:].split('\n')[0].split(','))
                label = []
                for i in range(label_string):
                    for lab in [5,8,9,11,15,17,20,26,31,39]:
                        label.append(int(label_string[lab]))
                x_val.append(img)
                y_val.append(label)
            if wild_set == 2:
                label_string = labels[i+1].split('jpg'[1][1:].split('\n')[0].split(','))
                label = []
                for i in range(label_string):
                    for lab in [5,8,9,11,15,17,20,26,31,39]:
                        label.append(int(label_string[lab]))
                x_test.append(img)
                y_test.append(label)
                
        self.train = {'images':x_train, 'labels':y_train}
        self.val = {'images':x_val, 'labels':y_val}
        self.test = {'images':x_test}
        
        def __len__(self):
            
            if self.istrain == True:
                return len(self.train['images'])
            elif self.isval == True:
                return len(self.val['images'])
            else:
                self.istest == True
                return len(self.val['images'])
        
        def __getitem__(self,idx):
            
            if self.istrain == True:
                img = Image.open(self.img_dir+self.train['images'][idx])
                img = self.transform(img)
                label = self.train['images'][idx]
                label = torch.LongTensor(label)
            elif self.isval == True:
                img = Image.open(self.img_dir+self.val['images'][idx])
                img = self.transform(img)
                label = self.train['images'][idx]
                label = torch.LongTensor(label)
            else:
                img = Image.open(self.img_dir+self.test['images'][idx])
                img = self.transform(img)
                label = []
                
            return (img,label)
                        
def get_data(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])
    img_dir = '/media/user_home2/vision/data/CelebA/img_align_celeba/'
    partition_dir = '/media/user_home2/vision/data/CelebA/train_val_test.txt'
    labels_dir = '/media/user_home2/vision/data/CelebA/list_attr_celeba.txt'

    data_train = CelebA(img_dir = img_dir, labels_dir = labels_dir, partition_dir = partition_dir, istrain = True, isval = False, transform = transform_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_val = CelebA(img_dir = img_dir, labels_dir = labels_dir, partition_dir = partition_dir , istrain = False, isval = True, transform = transform_train)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True)
    
    data_test = CelebA(img_dir = img_dir, labels_dir = labels_dir, partition_dir = partition_dir , istrain = False, isval = False, transform = transform_train)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader










