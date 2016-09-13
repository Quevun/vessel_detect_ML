# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:34:15 2016

@author: Quek Yu Yang
"""

import cv2
import numpy as np
import scipy.io
import featuremat
import os

if __name__ == "__main__":
    scales = np.arange(3,100,5)
    vessel_feature_mat = np.array([]).reshape(0,300)
    non_vessel_feature_mat = np.array([]).reshape(0,300)
    
    for filename in os.listdir('../data/vessels'):
        vessel_bin = np.load('../data/vessels/'+filename)
        img = cv2.imread('../data/color/'+filename.split('.')[0]+'.bmp')
        vessel_ind = np.nonzero(vessel_bin)
    
        feature_mat_maker = featuremat.FeatureMatMaker(img,scales)
        vessel_feature,non_vessel_feature = feature_mat_maker.getTrainMat(vessel_ind)
        vessel_feature_mat = np.concatenate((vessel_feature_mat,vessel_feature),0)
        non_vessel_feature_mat = np.concatenate((non_vessel_feature_mat,non_vessel_feature),0)
        
    """
    filename = 'tanaka2'
    vessel_bin = np.load('../data/vessels/'+filename+'.npy')
    img = cv2.imread('../data/color/'+filename+'.bmp')
    vessel_index = np.nonzero(vessel_bin)
    
    feature_mat_maker = featuremat.FeatureMatMaker(img,vessel_index,scales)
    vessel_feature_mat = feature_mat_maker.getTrainMat()   
    non_vessel_feature_mat = feature_mat_maker.getTrainMat(is_vessel=False)
    """
    
    scipy.io.savemat('../data/feature_mat/feature.mat',
                     dict(vessel_feature_mat=vessel_feature_mat,
                          non_vessel_feature_mat=non_vessel_feature_mat))