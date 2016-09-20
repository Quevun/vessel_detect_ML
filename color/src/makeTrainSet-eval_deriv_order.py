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
    vessel_feature_mat_order1 = np.array([]).reshape(0,9)
    non_vessel_feature_mat_order1 = np.array([]).reshape(0,9)
    vessel_feature_mat_order2 = np.array([]).reshape(0,18)
    non_vessel_feature_mat_order2 = np.array([]).reshape(0,18)
    vessel_feature_mat_order3 = np.array([]).reshape(0,30)
    non_vessel_feature_mat_order3 = np.array([]).reshape(0,30)
    
    for filename in os.listdir('../data/vessels/red'):
        vessel_bin = np.load('../data/vessels/red/'+filename)
        img = cv2.imread('../data/color/'+filename.split('.')[0]+'.bmp')
        img = cv2.pyrDown(img)
        vessel_ind = np.nonzero(vessel_bin)   
        feature_mat_maker = featuremat.FeatureMatMaker(img,scales)
        
        vessel_feature,non_vessel_feature = feature_mat_maker.getTrainMat(vessel_ind,order=1)
        vessel_feature_mat_order1 = np.concatenate((vessel_feature_mat_order1,vessel_feature),0)
        non_vessel_feature_mat_order1 = np.concatenate((non_vessel_feature_mat_order1,non_vessel_feature),0)
        
        vessel_feature,non_vessel_feature = feature_mat_maker.getTrainMat(vessel_ind,order=2)
        vessel_feature_mat_order2 = np.concatenate((vessel_feature_mat_order2,vessel_feature),0)
        non_vessel_feature_mat_order2 = np.concatenate((non_vessel_feature_mat_order2,non_vessel_feature),0)
        
        vessel_feature,non_vessel_feature = feature_mat_maker.getTrainMat(vessel_ind,order=3)
        vessel_feature_mat_order3 = np.concatenate((vessel_feature_mat_order3,vessel_feature),0)
        non_vessel_feature_mat_order3 = np.concatenate((non_vessel_feature_mat_order3,non_vessel_feature),0)

    scipy.io.savemat('../data/feature_mat/eval_deriv_order_feature.mat',
                     dict(vessel_feature_mat_order1=vessel_feature_mat_order1,
                          non_vessel_feature_mat_order1=non_vessel_feature_mat_order1,
                          vessel_feature_mat_order2=vessel_feature_mat_order2,
                          non_vessel_feature_mat_order2=non_vessel_feature_mat_order2,
                          vessel_feature_mat_order3=vessel_feature_mat_order3,
                          non_vessel_feature_mat_order3=non_vessel_feature_mat_order3))