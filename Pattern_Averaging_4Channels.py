# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:29:28 2020

@author: Chen Tuo
"""

from nd2reader import reader as nd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure
from scipy.ndimage import shift
import scipy.ndimage as ndimage
import time
from skimage.filters import threshold_triangle
from tifffile import imsave 

print('initializing...')

#Notes:
    #Please rename input images using continuous natural numbers starting from 0 (e.g., 0.nd2, 1.nd2, 2.nd2 ...) before running the script.
    #For convenience, the script only accepts images with the same number of pixels and Z slices. Please crop the images into squares.
    #If different Z stack numbers were used, please do max intensity projection manually and enter 1 for the parameter 'zstack'
    
start = time.time()

#Please set your parameters below

samplesize = 9 #Number of cells
imagesize = 500 #Dimensions of images
zstack = 1 #Number of Z slices
path = 'C:\Academic\Coding\maxproj' #Path to retrieve and save data 
project_name = 'outputfile' #Name to identify the output files

#Change this offset value if thresholding doesn't work well
offset = 0

print('...done, total time: %s s'%(time.time() - start))
print('converting images...')

#importing nd2 images and performing max intensity projection

image = np.zeros(imagesize*imagesize*4*zstack*samplesize).reshape(imagesize,imagesize,4,zstack,samplesize)
imagemax = np.zeros(imagesize*imagesize*4*samplesize).reshape(imagesize,imagesize,4,samplesize)

for a in range(samplesize):
    currentimage = nd.ND2Reader(path + '\%d.nd2'%(a))
    npimage = np.asarray(currentimage)
    rgb = np.zeros(npimage.shape[1]*npimage.shape[2]*4*zstack).reshape(npimage.shape[1],npimage.shape[2],4,zstack)
    for b in range(zstack):
        rgb[:,:,0,b] = nd.ND2Reader.get_frame_2D(currentimage,c=0,z=b)
        rgb[:,:,1,b] = nd.ND2Reader.get_frame_2D(currentimage,c=1,z=b)
        rgb[:,:,2,b] = nd.ND2Reader.get_frame_2D(currentimage,c=2,z=b)
        rgb[:,:,3,b] = nd.ND2Reader.get_frame_2D(currentimage,c=3,z=b)
    for c in range(npimage.shape[1]):
        for d in range(npimage.shape[2]):
            image[c,d,:,:,a] = rgb[c,d,:,:]
            for e in range(4):
                imagemax[c,d,e,a] = np.max(image[c,d,e,:,a])

print('...done, total time: %s s'%(time.time() - start))
print('looking for centroids...')

image_1 = imagemax[:,:,0,:]

mask = np.empty(image_1.shape)
rmask = np.empty(image_1.shape)
centroids = [] 

#thresholding 

for n in range(samplesize):
    thres = image_1[:,:,n] > threshold_triangle(image_1[:,:,n]) - offset*np.std(image_1[:,:,n])
    thres = ndimage.morphology.binary_fill_holes(thres)
    thres = ndimage.morphology.binary_closing(thres)
    plt.imshow(thres)
    mask[:,:,n] = thres
    
masksum = np.zeros(samplesize)

for n in range(samplesize):
    masksum[n] = np.sum(mask[:,:,n])
    
maskavg = np.sum(masksum)/samplesize
maskstd = np.std(masksum)

#detecting and fixing incomplete thresholding

for n in range(samplesize):
    m = 1
    while masksum[n] < maskavg-2*maskstd:
        mask[:,:,n] = image_1[:,:,n] > threshold_triangle(image_1[:,:,n]) - (offset+0.1*m)*np.std(image_1[:,:,n])
        mask[:,:,n] = ndimage.morphology.binary_fill_holes(mask[:,:,n])
        mask[:,:,n] = ndimage.morphology.binary_closing(mask[:,:,n])
        masksum[n] = np.sum(mask[:,:,n])
        maskavg = np.sum(masksum)/samplesize
        maskstd = np.std(masksum)
        m = m+1        

#calculating centroid coordinates

for n in range(samplesize):    
    label = measure.label(mask[:,:,n], connectivity=2)
    label_stats = measure.regionprops(label)
    largest_idx = np.argmax([aCell.area for aCell in label_stats])
    centroids.append(label_stats[largest_idx].centroid)
    [y, x] = label_stats[largest_idx].centroid
    plt.scatter(x, y, s=100, c='r')
    plt.imshow(mask[:,:,n])
    plt.show()
    rmask = 1 - mask

#masks with centroids will be shown in Plots

print('...done, total time: %s s'%(time.time() - start))
print('shifting...')

#create a placeholder for the corrected image

image_shifted = np.empty(image_1.shape)
image_shifted[:,:,0] = image_1[:,:,0]
image_shifted_2 = np.empty(image_1.shape)
image_shifted_2[:,:,0] = imagemax[:,:,1,0]
image_shifted_3 = np.empty(image_1.shape)
image_shifted_3[:,:,0] = imagemax[:,:,2,0]
image_shifted_4 = np.empty(image_1.shape)
image_shifted_4[:,:,0] = imagemax[:,:,3,0]

#shifting the images

for k in range(mask.shape[2]):
    first_cen_y = imagesize/2
    first_cen_x = imagesize/2
    next_cen_y, next_cen_x = centroids[k]
    image_shifted[:,:,k] = shift(image_1[:,:,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))
    image_shifted_2[:,:,k] = shift(imagemax[:,:,1,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))
    image_shifted_3[:,:,k] = shift(imagemax[:,:,2,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))
    image_shifted_4[:,:,k] = shift(imagemax[:,:,3,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))

print('...done, total time: %s s'%(time.time() - start))
print('overlaying...')

#overlaying different images

image_shifted_all = np.zeros(image.shape[0]*image.shape[1]*4).reshape(image.shape[0],image.shape[1],4)

for a in range(image.shape[0]):
    for b in range(image.shape[1]):
        image_shifted_all[a,b,0] = np.average(image_shifted[a,b,:])
        image_shifted_all[a,b,1] = np.average(image_shifted_2[a,b,:])
        image_shifted_all[a,b,2] = np.average(image_shifted_3[a,b,:])
        image_shifted_all[a,b,3] = np.average(image_shifted_4[a,b,:])

#saving each channel

image_display = image_shifted_all

plt.imshow(image_display[:,:,0]/65536)
plt.show()
plt.imshow(image_display[:,:,1]/65536)
plt.show()
plt.imshow(image_display[:,:,2]/65536)
plt.show()
plt.imshow(image_display[:,:,3]/65536)
plt.show()

print('...done, total time: %s s'%(time.time() - start))
print('saving data...')

imsave(path + '\\' + project_name + '_channel_1.tif', image_display[:,:,0].astype(np.uint16))
imsave(path + '\\' + project_name + '_channel_2.tif', image_display[:,:,1].astype(np.uint16))
imsave(path + '\\' + project_name + '_channel_3.tif', image_display[:,:,2].astype(np.uint16))
imsave(path + '\\' + project_name + '_channel_4.tif', image_display[:,:,3].astype(np.uint16))     

#calculating approximated average intensity for each channel within each cell

avg = np.zeros(4)
bg = np.zeros(4)
avgall = np.zeros(5*(samplesize+1)).reshape(5,samplesize+1)
bgall = np.zeros(5*(samplesize+1)).reshape(5,samplesize+1)
normalized = np.zeros(5*(samplesize+1)).reshape(5,samplesize+1)

for n in range(4):
    avg[n] = np.average(imagemax[:,:,n,:],weights = mask)
    bg[n] = np.average(imagemax[:,:,n,:],weights = rmask)
    for m in range(samplesize):
        avgall[n+1,m+1] = np.average(imagemax[:,:,n,m],weights = mask[:,:,m])
        bgall[n+1,m+1] = np.average(imagemax[:,:,n,m],weights = rmask[:,:,m])

normalized = avgall - bgall
result = pd.DataFrame(normalized)

for a in range(1,samplesize+1):
    result.iloc[0,a] = a-1

result.iloc[0,0] = 'Cell No.'
result.iloc[1,0] = 'Channel 1'
result.iloc[2,0] = 'Channel 2'
result.iloc[3,0] = 'Channel 3'
result.iloc[4,0] = 'Channel 4'

result.to_excel(path + '\\' + project_name + '_quantification.xlsx')

print('...all done, total time: %s s'%(time.time() - start))
