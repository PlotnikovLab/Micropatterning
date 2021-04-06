# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:29:28 2020

@author: Chen Tuo
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.ndimage as ndimage
from skimage import measure
from scipy.ndimage import shift
from skimage.filters import threshold_triangle
from tifffile import imsave 
from nd2reader import reader as nd

# Notes:
    # Please put all nd2 files to be analyzed in the same folder as the script before running, the output files will also be created in this folder.
    # For convenience, the script only accepts images with the same number of pixels and Z slices.
    # If different Z stack numbers were used for each sample, please do max intensity projection manually and enter 1 for the parameter 'zstack'
    # Identified cell shape with centroids will be shown in the Plots panel to review thresholding quality.

# ↓↓↓↓↓ Please set your parameters ↓↓↓↓↓

imagesize = 500 # Dimensions of images in pixels
zstack = 15 # Number of Z slices in each sample
thresholding_channel = 1 # The channel number used for segmenting cells (1-4)
optional_enabled = True # To enable additional output for further analyses, set this to True, otherwise set to False

offset = 0 # Slightly change this offset value only if thresholding doesn't work well (default = 0)
focalplane_channel = 2 # When optional features are enabled, set the channel number used for finding focal plane (1-4)
channel_of_interest = 4 # When optional features are enabled, set the channel number to be focused (1-4)

# ↑↑↑↑↑ Please set your parameters ↑↑↑↑↑

start = time.time()

# importing nd2 images and performing max intensity projection

path = os.getcwd()
samplesize = 0

for b,a in enumerate(os.listdir(path)):
    if a.endswith('.nd2'):
        samplesize = samplesize + 1
        
print('Number of samples: ' + str(samplesize))
print('Loading files...')

image = np.zeros(imagesize*imagesize*4*zstack*samplesize).reshape(imagesize,imagesize,4,zstack,samplesize)
imagemax = np.zeros(imagesize*imagesize*4*samplesize).reshape(imagesize,imagesize,4,samplesize)
samplenum = 0

for b,a in enumerate(os.listdir(path)):
    if a.endswith('.nd2'):
        currentimage = nd.ND2Reader(path + '\\' + a)
        rgb = np.zeros(imagesize*imagesize*4*zstack).reshape(imagesize,imagesize,4,zstack)
        for c in range(zstack):
            for d in range(4):
                rgb[:,:,d,c] = nd.ND2Reader.get_frame_2D(currentimage,c=d,z=b)
        for e in range(imagesize):
            for f in range(imagesize):
                image[e,f,:,:,samplenum] = rgb[e,f,:,:]
                for g in range(4):
                    imagemax[e,f,g,samplenum] = np.max(image[e,f,g,:,samplenum])
        samplenum = samplenum + 1

print('...done, total time: %s s'%(round(time.time() - start)))
print('Aligning cells...')

image_1 = imagemax[:,:,thresholding_channel - 1,:]

mask = np.empty(image_1.shape)
rmask = np.empty(image_1.shape)
centroids = [] 

# thresholding 

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

# detecting and fixing incomplete thresholding

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

# calculating centroid coordinates

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

# create a placeholder for the corrected image

image_shifted = np.empty(image_1.shape)
image_shifted[:,:,0] = imagemax[:,:,0,0]
image_shifted_2 = np.empty(image_1.shape)
image_shifted_2[:,:,0] = imagemax[:,:,1,0]
image_shifted_3 = np.empty(image_1.shape)
image_shifted_3[:,:,0] = imagemax[:,:,2,0]
image_shifted_4 = np.empty(image_1.shape)
image_shifted_4[:,:,0] = imagemax[:,:,3,0]

# shifting the images

for k in range(mask.shape[2]):
    first_cen_y = imagesize/2
    first_cen_x = imagesize/2
    next_cen_y, next_cen_x = centroids[k]
    image_shifted[:,:,k] = shift(imagemax[:,:,0,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))
    image_shifted_2[:,:,k] = shift(imagemax[:,:,1,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))
    image_shifted_3[:,:,k] = shift(imagemax[:,:,2,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))
    image_shifted_4[:,:,k] = shift(imagemax[:,:,3,k], shift=(-(next_cen_y-first_cen_y),-(next_cen_x-first_cen_x)))
    
print('...done, total time: %s s'%(round(time.time() - start)))
print('Generating averaged images...')

# overlaying different images

image_shifted_all = np.zeros(image.shape[0]*image.shape[1]*4).reshape(image.shape[0],image.shape[1],4)

for a in range(image.shape[0]):
    for b in range(image.shape[1]):
        image_shifted_all[a,b,0] = np.average(image_shifted[a,b,:])
        image_shifted_all[a,b,1] = np.average(image_shifted_2[a,b,:])
        image_shifted_all[a,b,2] = np.average(image_shifted_3[a,b,:])
        image_shifted_all[a,b,3] = np.average(image_shifted_4[a,b,:])

# saving each channel

image_display = image_shifted_all

plt.imshow(image_display[:,:,0]/65536)
plt.show()
plt.imshow(image_display[:,:,1]/65536)
plt.show()
plt.imshow(image_display[:,:,2]/65536)
plt.show()
plt.imshow(image_display[:,:,3]/65536)
plt.show()

imsave(path + '\\' + 'output' + '_channel_1.tif', image_display[:,:,0].astype(np.uint16))
imsave(path + '\\' + 'output' + '_channel_2.tif', image_display[:,:,1].astype(np.uint16))
imsave(path + '\\' + 'output' + '_channel_3.tif', image_display[:,:,2].astype(np.uint16))
imsave(path + '\\' + 'output' + '_channel_4.tif', image_display[:,:,3].astype(np.uint16))     

# calculating approximated average intensity for each channel within each cell

avg = np.zeros(4)
bg = np.zeros(4)
avgall = np.zeros(5*(samplesize+1)).reshape(5,samplesize+1)
bgall = np.zeros(5*(samplesize+1)).reshape(5,samplesize+1)
subtracted = np.zeros(5*(samplesize+1)).reshape(5,samplesize+1)

for n in range(4):
    avg[n] = np.average(imagemax[:,:,n,:],weights = mask)
    bg[n] = np.average(imagemax[:,:,n,:],weights = rmask)
    for m in range(samplesize):
        avgall[n+1,m+1] = np.average(imagemax[:,:,n,m],weights = mask[:,:,m])
        bgall[n+1,m+1] = np.average(imagemax[:,:,n,m],weights = rmask[:,:,m])

subtracted = avgall - bgall
result = pd.DataFrame(subtracted)

for a in range(1,samplesize+1):
    result.iloc[0,a] = a-1

result.iloc[0,0] = 'Cell No.'
result.iloc[1,0] = 'Channel 1'
result.iloc[2,0] = 'Channel 2'
result.iloc[3,0] = 'Channel 3'
result.iloc[4,0] = 'Channel 4'

result.to_excel(path + '\\' + 'output' + '_quantification.xlsx')

if optional_enabled == True:
    
    print('...done, total time: %s s'%(round(time.time() - start)))
    print('Normalizing...')
    
    # Normalizing each pixel to the average intensity within each cell
    
    normalized = np.zeros(imagesize*imagesize*4*samplesize).reshape(imagesize,imagesize,4,samplesize)
    normalized_averaged = np.zeros(imagesize*imagesize*4).reshape(imagesize,imagesize,4)
    
    normalized[:,:,0,:] = image_shifted
    normalized[:,:,1,:] = image_shifted_2
    normalized[:,:,2,:] = image_shifted_3
    normalized[:,:,3,:] = image_shifted_4
    
    for a in range(4):
        for b in range(samplesize):
            normalized[:,:,a,b] = normalized[:,:,a,b] / subtracted[a+1,b+1] * 1000
            
    # Creating averaged images using normalized individual images
    
    for a in range(imagesize):
        for b in range(imagesize):
            for c in range(4):
                normalized_averaged[a,b,c] = np.average(normalized[a,b,c,:])
                
    plt.imshow(normalized_averaged[:,:,0]/65536)
    plt.show()
    plt.imshow(normalized_averaged[:,:,1]/65536)
    plt.show()
    plt.imshow(normalized_averaged[:,:,2]/65536)
    plt.show()
    plt.imshow(normalized_averaged[:,:,3]/65536)
    plt.show()
    
    imsave(path + '\\' + 'output' + '_normalized_channel_1.tif', normalized_averaged[:,:,0].astype(np.uint16))
    imsave(path + '\\' + 'output' + '_normalized_channel_2.tif', normalized_averaged[:,:,1].astype(np.uint16))
    imsave(path + '\\' + 'output' + '_normalized_channel_3.tif', normalized_averaged[:,:,2].astype(np.uint16))
    imsave(path + '\\' + 'output' + '_normalized_channel_4.tif', normalized_averaged[:,:,3].astype(np.uint16)) 
    
    print('...done, total time: %s s'%(round(time.time() - start)))
    print('Focusing...')
    
    # Looking for the focal plane of each sample
    
    focalplane = np.zeros(samplesize)
    intensities = np.zeros(zstack*samplesize).reshape(zstack,samplesize)
    
    for a in range(zstack):
        for b in range(samplesize):
            intensities[a,b] = np.average(image[:,:,focalplane_channel - 1,a,b], weights = mask[:,:,b])
    
    for a in range(samplesize):
        focalplane = np.argmax(intensities, axis = focalplane_channel - 1)
        
    # Creating series of focused slices and averaged images
    
    exactfocus = np.zeros(imagesize*imagesize*samplesize).reshape(samplesize,imagesize,imagesize)
    exactfocusavg = np.zeros(imagesize*imagesize).reshape(imagesize,imagesize)
    exactfocusnor = np.zeros(imagesize*imagesize*samplesize).reshape(samplesize,imagesize,imagesize)
    exactfocusavgnor = np.zeros(imagesize*imagesize).reshape(imagesize,imagesize)
    
    for a in range(imagesize):
        for b in range(imagesize):
            for c in range(samplesize):
                exactfocus[c,a,b] = image[a,b,channel_of_interest - 1,focalplane[c],c]
    
    for a in range(imagesize):
        for b in range(imagesize):
            exactfocusavg[a,b] = np.average(exactfocus[:,a,b])
            
    for a in range(samplesize):
        exactfocusnor[a,:,:] = exactfocus[a,:,:] / subtracted[channel_of_interest,a+1] * 1000
        
    for a in range(imagesize):
        for b in range(imagesize):
            exactfocusavgnor[a,b] = np.average(exactfocusnor[:,a,b])
                
    imsave(path + '\\' + 'output' + '_focused_channel_' + str(channel_of_interest) + '.tif', exactfocus.astype(np.uint16), imagej = True)
    imsave(path + '\\' + 'output' + '_focused_overlay_channel_' + str(channel_of_interest) + '.tif', exactfocusavg.astype(np.uint16))
    imsave(path + '\\' + 'output' + '_focused_normalized_channel_' + str(channel_of_interest) + '.tif', exactfocusnor.astype(np.uint16), imagej = True)
    imsave(path + '\\' + 'output' + '_focused_overlay_normalized_channel_' + str(channel_of_interest) + '.tif', exactfocusavgnor.astype(np.uint16))
    
print('All done, total time: %s s'%(round(time.time() - start)))


