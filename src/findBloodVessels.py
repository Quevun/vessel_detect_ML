# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 15:58:10 2016

@author: Quek Yu Yang
"""

import scipy.io
import numpy as np
import cv2
import featuremat
import skimage.morphology
import morphology

def sigmoid(z):
    return 1/(1+np.exp(-z))

def clean(img):
    skel = skimage.morphology.skeletonize(img>0)
    branch_len = 20
    pruned = skel.astype(np.uint8)*255
    struc_ele1 = np.array([[-1,0,0],[1,1,0],[-1,0,0]])
    struc_ele2 = np.array([[1,0,0],[0,1,0],[0,0,0]])
    struc_ele_tuple = (struc_ele1,np.rot90(struc_ele1,1),np.rot90(struc_ele1,2),np.rot90(struc_ele1,3),
                       struc_ele2,np.rot90(struc_ele2,1),np.rot90(struc_ele2,2),np.rot90(struc_ele2,3))
    
    for i in range(branch_len):
        end_points = np.zeros(np.shape(skel)).astype(np.bool)
        for struc_ele in struc_ele_tuple:
            end_points = end_points + morphology.hitOrMiss(pruned,struc_ele)
        pruned = pruned * np.invert(end_points)
        
    struc_ele3 = np.array([[0,0,0],[0,1,0],[0,0,0]])
    single_points = morphology.hitOrMiss(pruned,struc_ele3)
    pruned = pruned * np.invert(single_points)
    return pruned
    
def findBloodVessels(img):
    scales = np.arange(3,25,5)
    feature_mat = featuremat.FeatureMatMaker(img,scales).getMat()
    mat_content = scipy.io.loadmat('../data/nn_param/major_vessels_only_7ppl-less_scales.mat')
    theta1 = mat_content['Theta1']
    theta2 = mat_content['Theta2']
    layer2_hypo = sigmoid(np.dot(feature_mat,theta1.T))
    
    temp = np.ones((np.size(layer2_hypo,0),np.size(layer2_hypo,1)+1))
    temp[:,1:] = layer2_hypo
    layer2_hypo = temp
    hypo = sigmoid(np.dot(layer2_hypo,theta2.T))
    predict = np.reshape(np.argmax(hypo,1),(img.shape[0],img.shape[1]),'F')
    predict = predict.astype(np.uint8)*255
    
    cleaned = clean(predict)
    return cleaned
    
if __name__ == '__main__':
    scales = np.arange(3,25,5)
    filename = 'kamiyama4'
    img = cv2.imread('../data/color/'+filename+'.bmp')
    img = cv2.pyrDown(img)
    feature_mat = featuremat.FeatureMatMaker(img,scales).getMat()
    mat_content = scipy.io.loadmat('../data/nn_param/major_vessels_only_7ppl.mat')
    theta1 = mat_content['Theta1']
    theta2 = mat_content['Theta2']
    layer2_hypo = sigmoid(np.dot(feature_mat,theta1.T))
    
    temp = np.ones((np.size(layer2_hypo,0),np.size(layer2_hypo,1)+1))
    temp[:,1:] = layer2_hypo
    layer2_hypo = temp
    hypo = sigmoid(np.dot(layer2_hypo,theta2.T))
    predict = np.reshape(np.argmax(hypo,1),(img.shape[0],img.shape[1]),'F')
    predict = predict.astype(np.uint8)*255
    
    #cv2.imwrite('../data/findBloodVessels_results/major_vessels_only_7ppl/'+filename+'.jpg',predict)
    cleaned = clean(predict)
    cv2.imwrite('../data/findBloodVessels_results/major_vessels_only_7ppl/'+filename+'.jpg',cleaned)
    #np.save('../data/findBloodVessels_results/major_vessels_only_7ppl/numpy_data/'+filename,predict)