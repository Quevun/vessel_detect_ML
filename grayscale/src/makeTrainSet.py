# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:34:15 2016

@author: Quek Yu Yang
"""

import cv2
import numpy as np
import hlpr
import scipy.io

def randNonVessel(shape,vessel_index):
    sample_size = np.size(vessel_index[0])
        
    y = vessel_index[0][np.newaxis,:]
    x = vessel_index[1][np.newaxis,:]
    vessel_ind_struc = np.concatenate((y,x),axis=0).flatten('F')
    vessel_ind_struc = vessel_ind_struc.view([('y',np.uint32),
                                              ('x',np.uint32)]) # structured array with (y,x) elements
    
    y = np.random.randint(shape[0],size=sample_size)[np.newaxis,:]
    x = np.random.randint(shape[1],size=sample_size)[np.newaxis,:]
    non_vessel_ind_struc = np.concatenate((y,x),axis=0).flatten('F')
    non_vessel_ind_struc = non_vessel_ind_struc.view([('y',np.uint32),
                                                      ('x',np.uint32)])
    non_vessel_ind_struc = np.unique(non_vessel_ind_struc)

    intersects = np.intersect1d(vessel_ind_struc,non_vessel_ind_struc,True)
    for intersect in intersects:
        non_vessel_ind_struc = np.delete(non_vessel_ind_struc, np.where(non_vessel_ind_struc==intersect))
    return non_vessel_ind_struc

def makeFeatureMatrix(img,index,scales):
    sample_size = np.size(index[0])
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
    feature_mat = np.zeros((sample_size,5*len(scales)))
    for i in range(len(v)):
        feature_mat[i,:] = v[i].flatten()
    return feature_mat

if __name__ == "__main__":
    scales = np.arange(3,200,5)
    vessel_bin = np.load('output/test_seven_vessels.npy')
    img = cv2.imread('input/IR3/test7.bmp',cv2.IMREAD_GRAYSCALE)
    vessel_index = np.nonzero(vessel_bin)
    vessel_feature_mat = makeFeatureMatrix(img,vessel_index,scales)  
    
    non_vessel_ind_struc = randNonVessel(img.shape,vessel_index)
    non_vessel_ind = (non_vessel_ind_struc['y'],non_vessel_ind_struc['x'])
    non_vessel_feature_mat = makeFeatureMatrix(img,non_vessel_ind,scales)
    scipy.io.savemat('feature_mat.mat',
                     dict(vessel_feature_mat=vessel_feature_mat,
                          non_vessel_feature_mat=non_vessel_feature_mat))