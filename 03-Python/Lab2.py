#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Laboratorio de Computer Vision
@author: Julio Nicolás Reyes
"""

##########################################################################################

#Importamos todas las librerias que necesitamos
import sys 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec
import numpy as np
from PIL import Image  
import urllib
import cv2
import zipfile
import time
import random

#Url en donde se encuentra la base de datos
url = 'https://www.dropbox.com/s/zyvhzzq4w1rhlvb/Dataset.zip?dl=1'

#Nombre del Dataset
Dataset = "Dataset.zip"

#Definimos el momento en el que se iniciará a contar el tiempo que dura el programa
now = time.time()

##########################################################################################

###########
#Para descargar imagen y guardarla en el computador
"""imagen = open("imagen.png",'wb')
imagen.write (urllib.urlopen(url).read())
imagen.close()"""


"""
url = urllib.request.urlopen('https://matplotlib.org/_images/stinkbug.png')
img = mpimg.imread('/Users/Nico/Desktop/Images/stinkbug.png')
img = mpimg.imread(url)
print(img)
img.shape"""

##########################################################################################

print('PARTE 1: Descarga de la base de datos')

print('La base de datos corresponde a Caltech 101. From: http://www.vision.caltech.edu/Image_Datasets/Caltech101/')

Read = urllib.urlopen(url) 
File=file(Dataset,"w")
File.write (Read.read())
File.close()
print('Se descargó la base de datos')

#Otra forma de descargar
#urllib.urlretrieve(url, "Dataset.zip") 

#Evaluamos el tiempo que demoró en descargar
Tiempo1 = time.time() - now

print "Descargado el archivo: %s en %0.3fs" % ("Dataset",Tiempo1)


##########################################################################################

print('PARTE 2: Descomprimir el archivo .zip descargado')

if (os.path.exists(Dataset)):
    FileDes = zipfile.ZipFile(Dataset, "r")
    FileDes.extractall('Dataset')
    FileDes.close()


##########################################################################################
print('PARTE 3: Visualizar las imágenes')

Dire = os.path.join('Dataset','Dataset') 
Dire2 = os.listdir(Dire)
fig, axs = plt.subplots(3, 4, figsize=(10, 10))

for v in range (1,13):
    x = random.randint(0,25) 
    
    if x == 19:
        Carp = os.path.join(Dire,Dire2[x+1])
        Carp2 = os.listdir(Carp)
        y = random.randint(0,30) 
        img= cv2.imread(os.path.join(Carp,Carp2[y]))
        img=cv2.resize(img,(256,256))
        
        a = plt.subplot(3, 4, v)
        
        plt.imshow(img)
        a.set_title(Carp[16:])
        plt.axis('off')
        
    else:
        Carp = os.path.join(Dire,Dire2[x])
        Carp2 = os.listdir(Carp)
        y = random.randint(0,30) 
        img= cv2.imread(os.path.join(Carp,Carp2[y]))
        img=cv2.resize(img,(256,256))
        
        a = plt.subplot(3, 4, v)
        
        plt.imshow(img)
        a.set_title(Carp[16:])
        plt.axis('off')
        

##########################################################################################
        
Tiempo2 = time.time() - now

print "El tiempo total de ejecución del programa fue de: %0.3fs" % (Tiempo2)



##########################################################################################

print('BIBLIOGRAFIA')

print('http://www.vision.caltech.edu/Image_Datasets/Caltech101/')
print('https://docs.python.org/2/library/urllib.html')
print('https://berkeley-stat159-f17.github.io/stat159-f17/lectures/10-matplotlib_beyond_basics/image_tutorial..html')
print('https://docs.python.org/2/library/os.path.html')
print('https://docs.python.org/2/library/os.html')
print('https://docs.python.org/2/library/urllib.html')
print('https://www.unioviedo.es/compnum/laboratorios_py/Intro_imagen/introduccion_imagen.html')
print('https://pythonprogramming.net/urllib-tutorial-python-3/')



