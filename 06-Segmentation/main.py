#!/usr/bin/python2
"""
Created on Thu Mar  7 22:06:31 2019

@author: Juan David Triana
"""

def groundtruth(img_file):
    import scipy.io as sio
    img = imageio.imread(img_file)
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,5][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')

def imshow(img, seg, title='Image'):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.imshow(seg, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    cb = plt.colorbar()
    cb.set_ticks(range(seg.max()+1))
    plt.title(title)
    plt.axis('off')
    plt.show()
#
def check_dataset(folder):
    import os
    from urllib import request
    import zipfile
    if not os.path.isdir(folder):
        # Download it.
        url = 'http://157.253.196.67/BSDS_small.zip'
        dataset = request.urlopen(url)
        data_read = dataset.read()
        file_s = open(os.getcwd()+'/'+'BSDS_small.zip','wb')
        file_s.write(data_read)
        file_s.close()
        
        zip_data = zipfile.Zipfile(os.getcwd()+'/'+'BSDS_small.zip','r')
        zip_data.extractall()
        zip_data.close()
        # Put your code here. Then remove the 'pass' command
    
if __name__ == '__main__':
    import argparse
    import imageio
    import time
    import segmentByClustering
    
    parser = argparse.ArgumentParser()
    now = time.time()
    
    #Numero de clusters
    k=9
    
    #Inicialización parámetros en la terminal
    parser.add_argument('--color', type=str, default='rgb', choices=['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']) # If you use more please add them to this list.
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--method', type=str, default='watershed', choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
    parser.add_argument('--img_file', type=str, required=True)
    #Funciones que sacan la segmentación de las imagenes. 
    
    #Observaciones:
    #Para kmeans k funciona bien entre: 7-10
    #Para watershed k funciona bien en 50
    #Para gmm k funciona bien en: 7
    #Para hierarchical k funciona bien en 20
    
    opts = parser.parse_args()
    check_dataset(opts.img_file.split('/'))
    
    img = imageio.imread(opts.img_file)
    seg = segmentByClustering(image=img, mode=opts.color, method=opts.method, k=opts.k)

    groundtruth(opts.img_file)
    
    Tiempo1 = time.time() - now
    print 'El tiempo total de ejecucion fue:', Tiempo1