# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 09:46:31 2016

@author: Quek Yu Yang
"""
import numpy as np
import cv2

class FeatureMat(object):
    def __init__(self,img,scales):
        assert np.size(img,2) == 3
        assert img.dtype == 'uint8'
        self.img = img
        self.scales = scales
        
    def getMat(self,index):
        sample_size = np.size(index[0])
        num_scales = len(self.scales)
        v = [np.zeros((num_scales,15)) for _ in xrange(sample_size)]
        
        for i in range(num_scales): 
            scaled = self.getScaledImg(self.img,self.scales[i])
            deriv_mat = self.getDerivMat(scaled,index)
            for j in range(sample_size):
                v[j][i,:] = deriv_mat[:,j]
        feature_mat = np.zeros((sample_size,15*num_scales))
        for i in range(len(v)):
            feature_mat[i,:] = v[i].flatten()
        return feature_mat
        
    def getDerivMat(self,scaled,index):
        first = scaled[:,:,0]   #first,second and third slice in axis 2
        second = scaled[:,:,1]
        third = scaled[:,:,2]
        
        x1 = cv2.Sobel(first,cv2.CV_64F,1,0)
        y1 = cv2.Sobel(first,cv2.CV_64F,0,1)
        xx1 = cv2.Sobel(first,cv2.CV_64F,2,0)
        yy1 = cv2.Sobel(first,cv2.CV_64F,0,2)
        xy1 = cv2.Sobel(first,cv2.CV_64F,1,1)
        x2 = cv2.Sobel(second,cv2.CV_64F,1,0)
        y2 = cv2.Sobel(second,cv2.CV_64F,0,1)
        xx2 = cv2.Sobel(second,cv2.CV_64F,2,0)
        yy2 = cv2.Sobel(second,cv2.CV_64F,0,2)
        xy2 = cv2.Sobel(second,cv2.CV_64F,1,1)
        x3 = cv2.Sobel(third,cv2.CV_64F,1,0)
        y3 = cv2.Sobel(third,cv2.CV_64F,0,1)
        xx3 = cv2.Sobel(third,cv2.CV_64F,2,0)
        yy3 = cv2.Sobel(third,cv2.CV_64F,0,2)
        xy3 = cv2.Sobel(third,cv2.CV_64F,1,1)
        
        ux1 = x1[index][np.newaxis,:] # Derivatives that correspond to coordinate of vessels
        uy1 = y1[index][np.newaxis,:]
        uxx1 = xx1[index][np.newaxis,:]
        uyy1 = yy1[index][np.newaxis,:]
        uxy1 = xy1[index][np.newaxis,:]
        ux2 = x2[index][np.newaxis,:]
        uy2 = y2[index][np.newaxis,:]
        uxx2 = xx2[index][np.newaxis,:]
        uyy2 = yy2[index][np.newaxis,:]
        uxy2 = xy2[index][np.newaxis,:]
        ux3 = x3[index][np.newaxis,:]
        uy3 = y3[index][np.newaxis,:]
        uxx3 = xx3[index][np.newaxis,:]
        uyy3 = yy3[index][np.newaxis,:]
        uxy3 = xy3[index][np.newaxis,:]
        
        deriv_mat = np.concatenate((ux1,uy1,uxx1,uyy1,uxy1,
                                    ux2,uy2,uxx2,uyy2,uxy2,
                                    ux3,uy3,uxx3,uyy3,uxy3),axis=0)
        return deriv_mat
        
    def getScaledImg(self,img,scale):
        sigma = np.sqrt(scale)
        size = int(np.ceil(sigma)*10+1)
        img = img.astype(np.float64)
        scaled_img = cv2.GaussianBlur(img,(size,size),sigma)
        return scaled_img