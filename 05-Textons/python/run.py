#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Julio Nicolas Reyes 
         Juan David Triana   
"""

import time
from sklearn.ensemble import RandomForestClassifier
import sys
import matplotlib.pyplot as plt

sys.path.append('python')

now = time.time()
#Se crea el banco de filtros
from fbCreate import fbCreate
bf = fbCreate(support = 2,startSigma = 0.6)


import numpy as np
import cifar10 as ci

#Se cargan las imagenes de la base de datos
x_im,y_im = ci.get_data(ci.load_cifar10(mode=1), sliced = 0.5)

#Se definen los clusters
kt = [32]
presicion = []
    
for k in kt:

    #Se corren los filtros del banco
    from fbRun import fbRun
    x_conc = np.reshape(x_im,[32,32*len(x_im)])
    respuestas_filtro = fbRun(bf,x_conc)
    
    print('Calculan los textones')
    #Se calculan los textones a partir de las respuestas
    from computeTextons import computeTextons
    map, textones = computeTextons(respuestas_filtro,k)
    
    #Se cargan las imagenes de Test
    
    
    #Se cargan las imagenes de Test
    x_imt,y_imt = ci.get_data(ci.load_cifar10(mode='test'), sliced = 0.1)
    
    #from skimage import color
    #from skimage import io
    
    
    
    #Se asignan los textones correspondientes a la base de Test
    from assignTextons import assignTextons
    
    #Funcion para hallar histogramas
    def histc(X, bins):
        import numpy as np
        map_to_bins = np.digitize(X,bins)
        r = np.zeros(bins.shape)
        for i in map_to_bins:
            r[i-1] += 1
        return np.array(r)
            
    train = []
    test = []
    his_train = []
    his_test = []
    
    print('Se asigan los textones a la base de train y se calculan sus histogramas')
    #Se asignan los textones a la base de train
    #for i in list(range(0,len(x_im))):
    #    train.append(np.expand_dims(assignTextons(fbRun(bf,x_im[i,::]),textones.transpose()),0)) 
    #print(len(train))
    for i in list(range(0,len(x_im))):
        mapa_train = (np.expand_dims(assignTextons(fbRun(bf,x_im[i,::]),textones.transpose()),0)) 
        his_train.append(histc(mapa_train.flatten(),np.arange(k))) 
    print(len(his_train))
    
    
    print('Se asigan los textones a la base de test y se calculan sus histogramas')
    for i in list(range(0,len(x_imt))):
        mapa_test = (np.expand_dims(assignTextons(fbRun(bf,x_imt[i,::]),textones.transpose()),0))
        his_test.append(histc(mapa_test.flatten(),np.arange(k))) 
    print(len(his_test))    
    
    
    print('Aplicamos el algoritmo de Ramdom Forest')    
    
    clf = RandomForestClassifier(n_estimators=100, random_state=40)
    
    #Construimos el "forest of trees" del conjunto de datos de entrenamiento (his_train, y_im).
    clf.fit(his_train, y_im)
    
    #Imprime las  caracteristicas mas importantes de cada clase
    print(clf.feature_importances_)
    
    #Miramos las prediciones en cada clase
    predicciones = clf.predict(his_test)
    
    i=0
    #Hallamos la precision promedio
    presicion.append(clf.score(his_test, y_imt))
    print('La presicion es de:', round(presicion[i]*100, 3), '%.')
    
    
    #Obtenemos la matriz de confusion
    print('Hallamos la matriz de confusion')
    #confusion_matrix(y_imt, predicciones)
    
    
    from MatrizCon import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_imt, predicciones)
    np.set_printoptions(precision=2)
    
    
    print('Matriz de confusion para K=', kt[i])
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                          title='Confusion matrix, without normalization')
    plt.show()
    
    plt.figure()
    b=plot_confusion_matrix(cnf_matrix, normalize=True, classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                          title='Confusion matrix, with normalization')
    plt.show()
    i=i+1
	
    print(presicion) 	    

    plt.figure()
    plt.bar(kt,presicion)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Accuracy of prediction %')
    plt.title('Accuracy vs Number of Clusters (k)')
    plt.show()

Tiempo1 = time.time() - now
print('El tiempo total de ejecucion fue:', Tiempo1)


    