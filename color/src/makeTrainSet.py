# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 14:34:15 2016

@author: Quek Yu Yang
"""

import cv2
import numpy as np
import scipy.io
import featuremat

if __name__ == "__main__":
    scales = np.arange(3,200,5)
    filename = 'tanaka2'
    vessel_bin = np.load('../data/vessels/'+filename+'.npy')
    img = cv2.imread('../data/color/'+filename+'.bmp')
    vessel_index = np.nonzero(vessel_bin)
    
    feature_mat_maker = featuremat.FeatureMatMaker(img,vessel_index,scales)
    vessel_feature_mat = feature_mat_maker.getMat()   
    non_vessel_feature_mat = feature_mat_maker.getMat(is_vessel=False)

    scipy.io.savemat('../data/feature_mat/'+filename+'.mat',
                     dict(vessel_feature_mat=vessel_feature_mat,
                          non_vessel_feature_mat=non_vessel_feature_mat))