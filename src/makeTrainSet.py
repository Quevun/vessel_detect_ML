# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:34:15 2016

@author: queky
"""

import cv2
import numpy as np
import hlpr
import scipy.io

scales = np.arange(3,200,5)
vessel_bin = cv2.imread('',cv2.IMREAD_GRAYSCALE)
non_vessel_bin = vessel_bin == 0
img = cv2.imread('',cv2.IMREAD_GRAYSCALE)

index = np.nonzero(vessel_bin)
sample_size = np.size(index[0])

########################################################
# Create a list of feature matrices for each sample

v = [np.zeros((len(scales),5)) for _ in xrange(sample_size)]


for i in range(len(scales)):
    scaled = hlpr.ScaledImage(img,scales[i])
    Ix = scaled.getDerivX()
    Iy = scaled.getDerivY()
    Ixx = scaled.getDerivXX()
    Iyy = scaled.getDerivYY()
    Ixy = scaled.getDerivXY()
    
    ux = Ix[index][np.newaxis,:] # Derivatives that correspond to coordinate of vessels
    uy = Iy[index][np.newaxis,:]
    uxx = Ixx[index][np.newaxis,:]
    uyy = Iyy[index][np.newaxis,:]
    uxy = Ixy[index][np.newaxis,:]

    temp = np.concatenate((ux,uy,uxx,uyy,uxy),axis=0)
    for j in range(sample_size):
        v[j][i,:] = temp[:,j]
        
for feature_vec in v:
    feature_vec = feature_vec.flatten()
    
scipy.io.savemat('feature_vectors.mat',dict(v=v))