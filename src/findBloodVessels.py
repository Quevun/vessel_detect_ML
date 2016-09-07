# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 15:58:10 2016

@author: Quek Yu Yang
"""

import scipy.io
import numpy as np
import cv2
import func

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
if __name__ == '__main__':
    scales = np.arange(3,200,5)
    img = cv2.imread('../data/IR4/yokoyama4.bmp',cv2.IMREAD_GRAYSCALE)
    feature_mat = func.extractFeatures(img,scales)
    mat_content = scipy.io.loadmat('../data/NN_param.mat')
    theta1 = mat_content['Theta1']
    theta2 = mat_content['Theta2']
    layer2_hypo = sigmoid(np.dot(feature_mat,theta1.T))
    
    temp = np.ones((np.size(layer2_hypo,0),np.size(layer2_hypo,1)+1))
    temp[:,1:] = layer2_hypo
    layer2_hypo = temp
    hypo = sigmoid(np.dot(layer2_hypo,theta2.T))
    predict = np.reshape(np.argmax(hypo,1),img.shape,'F')
    
    cv2.imwrite('../data/findBloodVessels_results/yokoyama4.jpg',predict.astype(np.uint8)*255)