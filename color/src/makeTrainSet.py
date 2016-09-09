# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:34:15 2016

@author: Quek Yu Yang
"""

import cv2
import numpy as np
import scipy.io
import featuremat

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

if __name__ == "__main__":
    scales = np.arange(3,200,5)
    vessel_bin = np.load('../data/vessels/tanaka1.npy')
    img = cv2.imread('../data/color/tanaka1.bmp')
    vessel_index = np.nonzero(vessel_bin)
    feature_mat_maker = featuremat.FeatureMatMaker(img,vessel_index,scales)
    vessel_feature_mat = feature_mat_maker.getMat()
    
    non_vessel_ind = feature_mat_maker.getMat(False)
    gray = cv2.imread('../data/IR/tanaka1.bmp',0)
    vessels = np.copy(gray)
    vessels[non_vessel_ind] = 255
    vessels = vessels*(vessel_bin==0)
    random = np.copy(gray)
    random[non_vessel_ind] = 255
    cv2.imwrite('../data/vessels.jpg',vessels)
    cv2.imwrite('../data/random.jpg',random)   #random intersects with vessels, needs fixing
    
    #non_vessel_ind = (non_vessel_ind_struc['y'],non_vessel_ind_struc['x'])
    #non_vessel_feature_mat = makeFeatureMatrix(img,non_vessel_ind,scales)
    #scipy.io.savemat('feature_mat.mat',
    #                 dict(vessel_feature_mat=vessel_feature_mat,
    #                      non_vessel_feature_mat=non_vessel_feature_mat))