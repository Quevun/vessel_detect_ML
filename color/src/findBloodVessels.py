# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 15:58:10 2016

@author: Quek Yu Yang
"""

import scipy.io
import numpy as np
import cv2
import featuremat

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
if __name__ == '__main__':
    scales = np.arange(3,100,5)
    filename = 'tanaka2'
    img = cv2.imread('../data/color/'+filename+'.bmp')
    feature_mat = featuremat.FeatureMatMaker(img,(1,1),scales).getMat()
    mat_content = scipy.io.loadmat('../data/nn_param/'+filename+'.mat')
    theta1 = mat_content['Theta1']
    theta2 = mat_content['Theta2']
    layer2_hypo = sigmoid(np.dot(feature_mat,theta1.T))
    
    temp = np.ones((np.size(layer2_hypo,0),np.size(layer2_hypo,1)+1))
    temp[:,1:] = layer2_hypo
    layer2_hypo = temp
    hypo = sigmoid(np.dot(layer2_hypo,theta2.T))
    predict = np.reshape(np.argmax(hypo,1),(img.shape[0],img.shape[1]),'F')
    
    cv2.imwrite('../data/findBloodVessels_results/'+filename+'.jpg',predict.astype(np.uint8)*255)