import sys
sys.path.append('python')

#Create a filter bank with deafult params
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization

#Load sample images from disk
from skimage import color
from skimage import io
from skimage.transform import resize

imBase1=color.rgb2gray(resize(io.imread('img/moto1.jpg'), (32, 32)))
imBase2=color.rgb2gray(resize(io.imread('img/perro1.jpg'), (32, 32)))

#Set number of clusters
k = 16*8

#Apply filterbank to sample image
from fbRun import fbRun
import numpy as np
filterResponses = fbRun(fb,np.hstack((imBase1,imBase2)))

#Computer textons from filter
from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)

#Load more images
imTest1=color.rgb2gray(resize(io.imread('img/moto2.jpg'), (32, 32)))
imTest2=color.rgb2gray(resize(io.imread('img/perro2.jpg'), (32, 32)))

#Calculate texton representation with current texton dictionary
from assignTextons import assignTextons
tmapBase1 = assignTextons(fbRun(fb,imBase1),textons.transpose())
tmapBase2 = assignTextons(fbRun(fb,imBase2),textons.transpose())
tmapTest1 = assignTextons(fbRun(fb,imTest1),textons.transpose())
tmapTest2 = assignTextons(fbRun(fb,imTest2),textons.transpose())

#Check the euclidean distances between the histograms and convince yourself that the images of the bikes are closer because they have similar texture pattern

# --> Can you tell why do we need to create a histogram before measuring the distance? <---

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

D = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
	 histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
D = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
	 histc(tmapTest2.flatten(), np.arange(k))/tmapTest2.size)

D = np.linalg.norm(histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size - \
	 histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
D = np.linalg.norm(histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size - \
	 histc(tmapTest2.flatten(), np.arange(k))/tmapTest2.size)