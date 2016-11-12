# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 15:58:10 2016

@author: Quek Yu Yang
"""

import scipy.io
import numpy as np
import cv2
import featuremat_noscale

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
if __name__ == '__main__':
    scales = np.arange(3,50,5)
    filename = 'quek1'
    img = cv2.imread('../data/color/'+filename+'.bmp')
    img = cv2.pyrDown(img)
    feature_mat = featuremat_noscale.FeatureMatMaker(img,scales).getMat()
    mat_content = scipy.io.loadmat('../data/nn_param/normalized_features_special.mat')
    theta1 = mat_content['Theta1']
    theta2 = mat_content['Theta2']
    layer2_hypo = sigmoid(np.dot(feature_mat,theta1.T))
    
    temp = np.ones((np.size(layer2_hypo,0),np.size(layer2_hypo,1)+1))
    temp[:,1:] = layer2_hypo
    layer2_hypo = temp
    hypo = sigmoid(np.dot(layer2_hypo,theta2.T))
    predict = np.reshape(np.argmax(hypo,1),(img.shape[0],img.shape[1]),'F')
    
    #hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #skin_bin = segmentSkin(hsv)
    #cv2.imshow('stuff',skin_bin.astype(np.uint8)*255)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    cv2.imwrite('../data/findBloodVessels_results/noscale/'+filename+'.jpg',predict.astype(np.uint8)*255)