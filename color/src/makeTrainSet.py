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
    scales = np.arange(3,50,5)
    num_features = len(scales)*30
    threshold = 30000
    vessel_feature_mat = np.array([]).reshape(0,num_features)
    non_vessel_feature_mat = np.array([]).reshape(0,num_features)
    
    i = 1
    for filename in os.listdir('../data/vessels/major_vessels_only'):
        vessel_bin = np.load('../data/vessels/major_vessels_only/'+filename)
        img = cv2.imread('../data/color/'+filename.split('.')[0]+'.bmp')
        img = cv2.pyrDown(img)
        vessel_ind = np.nonzero(vessel_bin)
    
        feature_mat_maker = featuremat.FeatureMatMaker(img,scales)
        vessel_feature,non_vessel_feature = feature_mat_maker.getTrainMat(vessel_ind)
        vessel_feature_mat = np.concatenate((vessel_feature_mat,vessel_feature),0)
        non_vessel_feature_mat = np.concatenate((non_vessel_feature_mat,non_vessel_feature),0)
        
        sample_size = vessel_feature_mat.shape[0]+non_vessel_feature_mat.shape[0]
        if sample_size > threshold:    
            scipy.io.savemat('../data/feature_mat/major_vessels_only_7ppl/batch'+str(i)+'.mat',
                             dict(vessel_feature_mat=vessel_feature_mat,
                                  non_vessel_feature_mat=non_vessel_feature_mat))
            i += 1
            vessel_feature_mat = np.array([]).reshape(0,num_features)
            non_vessel_feature_mat = np.array([]).reshape(0,num_features)
        
    if vessel_feature_mat.size != 0:
        scipy.io.savemat('../data/feature_mat/major_vessels_only_7ppl/batch'+str(i)+'.mat',
                             dict(vessel_feature_mat=vessel_feature_mat,
                                  non_vessel_feature_mat=non_vessel_feature_mat))