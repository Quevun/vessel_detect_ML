# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 11:11:37 2016

@author: keisoku
"""
import cv2
import numpy as np
import featuremat

"""#test FeatureMat
vessel_bin = np.load('../data/vessels/tanaka1.npy')
vessel_index = np.nonzero(vessel_bin)
scales = np.arange(3,200,5)
img = cv2.imread('../data/color/tanaka1.bmp')

###############################################
#   vessels

color_feature_mat = featuremat.FeatureMatMaker(img,vessel_index,scales).getMat()
gray_feature_mat1 = makeTrainSet2.makeFeatureMatrix(img[:,:,0],vessel_index,scales)
gray_feature_mat2 = makeTrainSet2.makeFeatureMatrix(img[:,:,1],vessel_index,scales)
gray_feature_mat3 = makeTrainSet2.makeFeatureMatrix(img[:,:,2],vessel_index,scales)

for i in range(len(scales)):
    print np.array_equal(color_feature_mat[:,i*15:i*15+5],gray_feature_mat1[:,i*5:i*5+5])
    print np.array_equal(color_feature_mat[:,i*15+5:i*15+5+5],gray_feature_mat2[:,i*5:i*5+5])
    print np.array_equal(color_feature_mat[:,i*15+10:i*15+10+5],gray_feature_mat3[:,i*5:i*5+5])
##############################################


##############################################
#   non-vessels

color_feature_mat,non_vessel_ind = featuremat.FeatureMatMaker(img,vessel_index,scales).getMat(False)
gray_feature_mat1 = makeTrainSet2.makeFeatureMatrix(img[:,:,0],non_vessel_ind,scales)
gray_feature_mat2 = makeTrainSet2.makeFeatureMatrix(img[:,:,1],non_vessel_ind,scales)
gray_feature_mat3 = makeTrainSet2.makeFeatureMatrix(img[:,:,2],non_vessel_ind,scales)

for i in range(len(scales)):
    print np.array_equal(color_feature_mat[:,i*15:i*15+5],gray_feature_mat1[:,i*5:i*5+5])
    print np.array_equal(color_feature_mat[:,i*15+5:i*15+5+5],gray_feature_mat2[:,i*5:i*5+5])
    print np.array_equal(color_feature_mat[:,i*15+10:i*15+10+5],gray_feature_mat3[:,i*5:i*5+5])
#############################################
"""


# test FeatureMat.getTraintMat
scales = np.arange(3,100,5)
img = cv2.imread('../data/color/tanaka2.bmp')
vessel_bin = np.load('../data/vessels/tanaka2.npy')
vessel_ind = np.nonzero(vessel_bin)
featuremat_maker = featuremat.FeatureMatMaker(img,scales)
vessel_feature,non_vessel_feature = featuremat_maker.getTrainMat(vessel_ind)