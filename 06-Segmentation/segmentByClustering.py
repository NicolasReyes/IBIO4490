#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 00:24:13 2019

"""

import scipy.io as sio
import matplotlib.pyplot as plt
import imageio
import numpy as np
from skimage.color import rgb2gray
from skimage import io, util, filters, color
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from sklearn import mixture
from skimage.segmentation import mark_boundaries
from skimage.data import coffee
from skimage.util import img_as_float
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


#Funcion para normalizar los espacios de color
def debugImg(image):
  import cv2
  normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  return normalized_image

def evaluation(img_file,predict):
    import numpy as np
    image= imageio.imread(img_file)
    nx, ny, canal = image.shape
    gt = sio.loadmat(img_file.replace('jpg', 'mat'))
    grounds = {'G1':np.zeros(shape=(nx,ny),dtype=np.uint16),'G2':np.zeros(shape=(nx,ny),dtype=np.uint16),'G3':np.zeros(shape=(nx,ny),dtype=np.uint16),'G4':np.zeros(shape=(nx,ny),dtype=np.uint16),'G5':np.zeros(shape=(nx,ny),dtype=np.uint16)}
    dices = np.zeros(shape = (1,5),dtype = float)
    for i in range(0,4):
        segm = gt['groundTruth'][0,i][0][0]['Segmentation']
        
        grounds['G'+str(i+1)] = segm
        intersection = np.logical_and(predict,segm)
        intersection_sum = float(sum(sum(intersection)))
        total_area = float(nx*ny)
        #import pdb; pdb.set_trace()
        dice = intersection_sum/total_area
        dices[0,i] = dice
            
    score = np.mean(dices)
    return score

#Se definen los diferentes espacios de color
def colorSpace(image,mode):
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
    
    image = io.imread(image)
    (x, y, depth) = image.shape
    #image = img_as_float(image()[::2, ::2])#Para volverla tipo float y reducir su tamaño a la mitad
    
    if mode == 'lab':
        image = color.rgb2lab(image)
        image = debugImg(image)
        plt.figure()
        plt.imshow(image)
        plt.title("Color Space: lab", fontdict=font)
        
    elif mode == 'rgb':
        image = debugImg(image)
        plt.figure()
        plt.imshow(image)
        plt.title("Color Space: rgb", fontdict=font)
        
    elif mode == 'xyz':
        image = color.rgb2xyz(image)
        image = debugImg(image)
        plt.figure()
        plt.imshow(image)
        plt.title("Color Space: xyz", fontdict=font)
        
    elif mode == 'rgb+xy':    

        x1 = np.zeros((x, y, 1))
        y1 = np.zeros((x, y, 1))
        
        for i in range(x):
            x1[i, :, :]+=i
            
        for j in range(y):
            y1[:, j, :]+=j
        image = np.append(image, x1, axis=2)
        image = np.append(image, y1, axis=2)
        #print(image) 
        
    elif mode == 'lab+xy':  
        image = color.rgb2lab(image)

        x1 = np.zeros((x, y, 1))
        y1 = np.zeros((x, y, 1))
        
        for i in range(x):
            x1[i, :, :]+=i
            
        for j in range(y):
            y1[:, j, :]+=j
        image = np.append(image, x1, axis=2)
        image = np.append(image, y1, axis=2)
        #debugImg(image)
        #plt.title("Color Space: lab+xy", fontdict=font)
        
    elif mode == 'hsv+xy':  
        image = color.rgb2hsv(image)
        image = debugImg(image)

        x1 = np.zeros((x, y, 1))
        y1 = np.zeros((x, y, 1))
        
        for i in range(x):
            x1[i, :, :]+=i
            
        for j in range(y):
            y1[:, j, :]+=j
        image = np.append(image, x1, axis=2)
        image = np.append(image, y1, axis=2)
        #plt.figure()
        #plt.imshow(image)
        #plt.title("Color Space: hsv+xy", fontdict=font)
        
    else:
        mode = 'hsv'
        image = color.rgb2hsv(image)
        image = debugImg(image)
        plt.figure()
        plt.imshow(image)
        plt.title("Color Space: hsv", fontdict=font)
        
    return image
        


#Funcion para definir el metodo de clustering
def Method(image,method,k):
        
        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
        
        #image = io.imread(image)
        (x, y, depth) = image.shape
        image2 = image.reshape((x*y, depth))
        
        """
        if method == 'kmeans':
            #segments = slic(image, n_segments=40, compactness=10)
            segments = slic(image, n_segments=k, compactness=3, sigma=1)
        
            plt.figure()
            #plt.imshow(mark_boundaries(image, segments), cmap=plt.get_cmap('tab20b'))
            plt.imshow(segments, cmap=plt.get_cmap('tab20b'))
            plt.title("Segment by K-means", fontdict=font)
            """
        if method == 'kmeans':
            
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans = kmeans.fit(image2)
            predict = kmeans.predict(image2).reshape(x, y)
            plt.figure()
            plt.imshow(predict)
            plt.title("Segment by kmeans", fontdict=font)
            
        elif method == 'watershed':
            borde = filters.sobel(rgb2gray(image))
            predict = watershed(borde, markers=k, compactness=0.001)
            
            plt.figure()
            plt.imshow(mark_boundaries(image, predict), cmap=plt.get_cmap('spring'))
            plt.title("Segment by Watershed", fontdict=font)
            #return im 
            
        elif method == 'gmm':
            #gmm = GaussianMixture(n_components=1, covariance_type=’full’, tol=0.001, reg_covar=1e-06, max_iter=100)
            gmm = mixture.GaussianMixture(n_components=k, covariance_type='full')
            gmm=gmm.fit(image2)
            predict = gmm.predict(image2).reshape(x, y)
            plt.figure()
            plt.imshow(predict, cmap=plt.get_cmap('coolwarm'))
            plt.title("Segment by gmm", fontdict=font)
            
        else:    
            method == 'hierarchical'
            connectivity = grid_to_graph(image.shape[0], image.shape[1])
            
            ward = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=connectivity)
            ward.fit(image2)
            predict = ward.labels_.astype(np.int).reshape(x, y)
            plt.figure()
            plt.imshow(predict)
            plt.title("Segment by Hierarchical", fontdict=font)
        return predict
    
#Definimos la funcion que unifica los espacios de color y el metodo de clustering            
def segmentByClustering(image, mode, method, k):
    image = colorSpace(image, mode)
    Segmen = Method(image, method, k)
    return Segmen    