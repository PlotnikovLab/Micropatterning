# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:29:28 2020

@author: polar
"""

from nd2reader import reader as nd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio
from skimage import io
from skimage import measure
from scipy.ndimage import shift
import time
from skimage.filters import threshold_triangle
from tifffile import imsave 

print('initializing...')

start = time.time()

samplesize = 9
imagesize = 500
zstack = 1
path = 'C:\Academic\Coding\CCCC' 
output = 'outputfile'

image = np.zeros(imagesize*imagesize*3*zstack*samplesize).reshape(imagesize,imagesize,3,zstack,samplesize)
imagemax = np.zeros(imagesize*imagesize*3*samplesize).reshape(imagesize,imagesize,3,samplesize)

print('...done, total time: %s s'%(time.time() - start))
print('converting images...')

for a in range(samplesize):
    currentimage = nd.ND2Reader('C:\Academic\Coding\maxproj\%d.nd2'%(a))
    npimage = np.asarray(currentimage)
    rgb = np.zeros(npimage.shape[1]*npimage.shape[2]*3*zstack).reshape(npimage.shape[1],npimage.shape[2],3,zstack)
    for b in range(zstack):
        rgb[:,:,0,b] = nd.ND2Reader.get_frame_2D(currentimage,c=0,z=b)
        rgb[:,:,1,b] = nd.ND2Reader.get_frame_2D(currentimage,c=2,z=b)
        rgb[:,:,2,b] = nd.ND2Reader.get_frame_2D(currentimage,c=1,z=b)
    for c in range(npimage.shape[1]):
        for d in range(npimage.shape[2]):
            image[c,d,:,:,a] = rgb[c,d,:,:]
            for e in range(3):
                imagemax[c,d,e,a] = np.max(image[c,d,e,:,a])

print('...done, total time: %s s'%(time.time() - start))
print('looking for centroids...')

for a in range(samplesize):
    plt.imshow(imagemax[:,:,:,a]/65536)
    plt.show()

image_1 = imagemax[:,:,0,:]

mask = np.empty(image_1.shape)
rmask = np.empty(image_1.shape)
centroids = [] 

for n in range(image_1.shape[2]):
    #thresholding
    #thres = image_1[:,:,n] > np.mean(image_1[:,:,n]) + 0.1*np.std(image_1[:,:,n]) 
    thres = image_1[:,:,n] > threshold_triangle(image_1[:,:,n])
    plt.imshow(thres)
    #filling the mask
    mask[:,:,n] = thres
    #label the mask
    label = measure.label(thres, connectivity=2)
    label_stats = measure.regionprops(label)
    #getting the index of the largest object
    largest_idx = np.argmax([aCell.area for aCell in label_stats])
    centroids.append(label_stats[largest_idx].centroid)
    #saving the centroids
    [y, x] = label_stats[largest_idx].centroid
    plt.scatter(x, y, s=100, c='r')
    plt.show()
    rmask = 1 - mask

print('...done, total time: %s s'%(time.time() - start))
print('shifting...')

#create a placeholder for the corrected image
image_shifted = np.empty(image_1.shape)
image_shifted[:,:,0] = image_1[:,:,0]
#create a placeholder for the corrected image
image_shifted_2 = np.empty(image_1.shape)
image_shifted_2[:,:,0] = imagemax[:,:,1,0]
image_shifted_3 = np.empty(image_1.shape)
image_shifted_3[:,:,0] = imagemax[:,:,2,0]

#shifting the images
for k in range(0,mask.shape[2]):
    #first_cen_y, first_cen_x = centroids[0]
    first_cen_y = 290
    first_cen_x = 250
    next_cen_y, next_cen_x = centroids[k]
    image_shifted[:,:,k] = shift(image_1[:,:,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))
    image_shifted_2[:,:,k] = shift(imagemax[:,:,1,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))
    image_shifted_3[:,:,k] = shift(imagemax[:,:,2,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))

#saving the images

print('...done, total time: %s s'%(time.time() - start))
print('overlaying...')

image_shifted_all = np.zeros(image.shape[0]*image.shape[1]*3).reshape(image.shape[0],image.shape[1],3)

for a in range(image.shape[0]):
    for b in range(image.shape[1]):
        image_shifted_all[a,b,0] = np.average(image_shifted[a,b,:])
        image_shifted_all[a,b,1] = np.average(image_shifted_2[a,b,:])
        image_shifted_all[a,b,2] = np.average(image_shifted_3[a,b,:])

plt.imshow(image_shifted_all/65536)
plt.show()

test = image_shifted_all * 50
plt.imshow(test/65536)
plt.show()

imsave('aligned.tif', image_shifted_all.astype(np.uint16))
imsave('adjusted.tif', test.astype(np.uint16))

print('...done, total time: %s s'%(time.time() - start))
print('calculating...')     

avg = np.zeros(3)
bg = np.zeros(3)
avgall = np.zeros(3*samplesize).reshape(3,samplesize)
bgall = np.zeros(3*samplesize).reshape(3,samplesize)
normalized = np.zeros(3*samplesize).reshape(3,samplesize)

for n in range(3):
    avg[n] = np.average(imagemax[:,:,n,:],weights = mask)
    bg[n] = np.average(imagemax[:,:,n,:],weights = rmask)
    for m in range(samplesize):
        avgall[n,m] = np.average(imagemax[:,:,n,m],weights = mask[:,:,m])
        bgall[n,m] = np.average(imagemax[:,:,n,m],weights = rmask[:,:,m])
    
print('Actin\IL-1\DAPI, average & background')

normalized = avgall - bgall
result = pd.DataFrame(normalized)

result.to_excel('actmyo.xlsx')

print('...all done, total time: %s s'%(time.time() - start))

