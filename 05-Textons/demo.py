#!/usr/bin/python2
"""
Created on Thu Feb 28 22:05:45 2019

@author: Juan David Triana
         Nicolas Reyes
"""
import numpy as np
import cifar10 as ci

#Se carga el banco de filtros
from fbCreate import fbCreate
bf = fbCreate(support = 2,startSigma = 0.6)

#Se carga la base de datos de test
test_im, test_label = ci.get_data(ci.load_cifar10(mode='test'),sliced=0.01)

#Se corren los filtros del banco
from fbRun import fbRun
x_conc = np.reshape(test_im,[32,32*len(test_im)])
respuestas_filtro = fbRun(bf,x_conc)

k = 42
#Se calculan los textones a partir de las respuestas
from computeTextons import computeTextons
map, textones = computeTextons(respuestas_filtro,k)

#Funcion para hallar histogramas
def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

test = []
hists = []
from assignTextons import assignTextons
print('Se asigan los textones a la base de test y se calculan sus histogramas')
for i in list(range(0,len(test_im))):
    mapa_test = (np.expand_dims(assignTextons(fbRun(bf,test_im[i,::]),textones.transpose()),0))
    hists.append(histc(mapa_test.flatten(),np.arange(k)))    

#Cargar el modelo ya entrenado
import pickle

pkl_file = "RF_model.pkl"

with open(pkl_file,'rb') as file:
    pickle_model = pickle.load(file)
    
#Calcula los valore a predecir de acuerdo al modelo de clasificacion

prediction = pickle_model.predict(hists)

import matplotlib.pyplot as plt
import matplotlib.image as img
import random as rn

imagenRandom = rn.choice(test_im)
plt.axis('off')
plt.imshow(imagenRandom)
labelRandom = rn.choice(test_label)
print labelRandom

















