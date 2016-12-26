# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 09:46:31 2016

@author: Quek Yu Yang
"""
import numpy as np
import cv2
import psutil
import sys

class FeatureMatMaker(object):
    def __init__(self,img,scales):
        assert np.size(img,2) == 3
        assert img.dtype == 'uint8'
        self.img = img
        self.scales = scales
        self.num_features = 20
        
    def getMat(self):
        hsv = cv2.cvtColor(self.img,cv2.COLOR_BGR2HSV)
        nf = self.num_features   # Short for number of features
        shape = (np.size(self.img,0),np.size(self.img,1),len(self.scales)*nf+1)
        feature_cube = np.zeros(shape)
        for i in range(len(self.scales)):
            scaled = getScaledImg(hsv,self.scales[i])
            first = scaled[:,:,0]
            second = scaled[:,:,1]
            
            x1 = cv2.Sobel(first,cv2.CV_64F,1,0)[:,:,np.newaxis]
            y1 = cv2.Sobel(first,cv2.CV_64F,0,1)[:,:,np.newaxis]
            xx1 = cv2.Sobel(first,cv2.CV_64F,2,0)[:,:,np.newaxis]
            yy1 = cv2.Sobel(first,cv2.CV_64F,0,2)[:,:,np.newaxis]
            xy1 = cv2.Sobel(first,cv2.CV_64F,1,1)[:,:,np.newaxis]
            xxx1 = cv2.Sobel(first,cv2.CV_64F,3,0,ksize=5)[:,:,np.newaxis]
            xxy1 = cv2.Sobel(first,cv2.CV_64F,2,1,ksize=5)[:,:,np.newaxis]
            xyy1 = cv2.Sobel(first,cv2.CV_64F,1,2,ksize=5)[:,:,np.newaxis]
            yyy1 = cv2.Sobel(first,cv2.CV_64F,0,3,ksize=5)[:,:,np.newaxis]
            
            x2 = cv2.Sobel(second,cv2.CV_64F,1,0)[:,:,np.newaxis]
            y2 = cv2.Sobel(second,cv2.CV_64F,0,1)[:,:,np.newaxis]
            xx2 = cv2.Sobel(second,cv2.CV_64F,2,0)[:,:,np.newaxis]
            yy2 = cv2.Sobel(second,cv2.CV_64F,0,2)[:,:,np.newaxis]
            xy2 = cv2.Sobel(second,cv2.CV_64F,1,1)[:,:,np.newaxis]
            xxx2 = cv2.Sobel(second,cv2.CV_64F,3,0,ksize=5)[:,:,np.newaxis]
            xxy2 = cv2.Sobel(second,cv2.CV_64F,2,1,ksize=5)[:,:,np.newaxis]
            xyy2 = cv2.Sobel(second,cv2.CV_64F,1,2,ksize=5)[:,:,np.newaxis]
            yyy2 = cv2.Sobel(second,cv2.CV_64F,0,3,ksize=5)[:,:,np.newaxis]
            
            first = first[:,:,np.newaxis]
            second = second[:,:,np.newaxis]
            feature_cube[:,:,nf*i+1:nf*i+nf+1] = np.concatenate((first,x1,y1,xx1,yy1,xy1,xxx1,xxy1,xyy1,yyy1,
                                                                 second,x2,y2,xx2,yy2,xy2,xxx2,xxy2,xyy2,yyy2),axis=2)
        
        feature_mat = np.reshape(feature_cube,(shape[0]*shape[1],shape[2]),'F')
        feature_mat = self.featureScale(feature_mat)
        feature_mat[:,0] = 1  # First element of each feature vector is 1
        return feature_mat
            
    def getTrainMat(self,vessel_ind):
        
        #######################
        #   Initialization
        hsv = cv2.cvtColor(self.img,cv2.COLOR_BGR2HSV)
        self.vessel_ind = vessel_ind
        self.vessel_sample_size = self.vessel_ind[0].size
        num_scales = len(self.scales)
        vessel_v = [np.zeros((num_scales,self.num_features)) for _ in xrange(self.vessel_sample_size)]
        
        non_vessel_ind = self.getRandInd()
        non_vessel_sample_size = non_vessel_ind[0].size
        non_vessel_v = [np.zeros((num_scales,self.num_features)) for _ in xrange(non_vessel_sample_size)]
        #######################        
        
        for i in range(num_scales): 
            scaled = getScaledImg(hsv,self.scales[i])
            vessel_deriv_mat,non_vessel_deriv_mat = self.getDerivMat(scaled,self.vessel_ind,
                                                                     non_vessel_ind)
            for j in range(self.vessel_sample_size):
                vessel_v[j][i,:] = vessel_deriv_mat[:,j]
            for j in range(non_vessel_sample_size):
                non_vessel_v[j][i,:] = non_vessel_deriv_mat[:,j]
                
        #######################
        #   Feature matrix
        vessel_feature_mat = np.zeros((self.vessel_sample_size,self.num_features*num_scales))
        non_vessel_feature_mat = np.zeros((non_vessel_sample_size,self.num_features*num_scales))
        for i in range(len(vessel_v)):
            vessel_feature_mat[i,:] = vessel_v[i].flatten()
        for i in range(len(non_vessel_v)):
            non_vessel_feature_mat[i,:] = non_vessel_v[i].flatten()
        ######################
            
        scaled_features = self.featureScale(np.concatenate((vessel_feature_mat,
                                                           non_vessel_feature_mat),axis=0))
        vessel_feature_mat = scaled_features[:self.vessel_sample_size,:]
        non_vessel_feature_mat = scaled_features[-non_vessel_sample_size:,:]

        return vessel_feature_mat,non_vessel_feature_mat
            
    def getRandInd(self):
        y = self.vessel_ind[0][np.newaxis,:]
        x = self.vessel_ind[1][np.newaxis,:]
        vessel_ind_struc = np.concatenate((y,x),axis=0).flatten('F')
        vessel_ind_struc = vessel_ind_struc.view([('y',np.int64),
                                                  ('x',np.int64)]) # structured array with (y,x) elements
        y = np.random.randint(self.img.shape[0],size=self.img.shape[0]*self.img.shape[1]/16)[np.newaxis,:]
        x = np.random.randint(self.img.shape[1],size=self.img.shape[0]*self.img.shape[1]/16)[np.newaxis,:]
        non_vessel_ind_struc = np.concatenate((y,x),axis=0).flatten('F')
        non_vessel_ind_struc = non_vessel_ind_struc.astype(np.int64)
        non_vessel_ind_struc = non_vessel_ind_struc.view([('y',np.int64),
                                                          ('x',np.int64)])
        non_vessel_ind_struc = np.unique(non_vessel_ind_struc)
    
        intersects = np.intersect1d(vessel_ind_struc,non_vessel_ind_struc,True)
        for intersect in intersects:
            non_vessel_ind_struc = np.delete(non_vessel_ind_struc, np.where(non_vessel_ind_struc==intersect))
        non_vessel_ind = (non_vessel_ind_struc['y'],non_vessel_ind_struc['x'])
        return non_vessel_ind
        
    def getDerivMat(self,scaled,vessel_ind,non_vessel_ind):
        first = scaled[:,:,0]
        second = scaled[:,:,1]
        
        x1 = cv2.Sobel(first,cv2.CV_64F,1,0)
        y1 = cv2.Sobel(first,cv2.CV_64F,0,1)
        xx1 = cv2.Sobel(first,cv2.CV_64F,2,0)
        yy1 = cv2.Sobel(first,cv2.CV_64F,0,2)
        xy1 = cv2.Sobel(first,cv2.CV_64F,1,1)
        xxx1 = cv2.Sobel(first,cv2.CV_64F,3,0,ksize=5)
        xxy1 = cv2.Sobel(first,cv2.CV_64F,2,1,ksize=5)
        xyy1 = cv2.Sobel(first,cv2.CV_64F,1,2,ksize=5)
        yyy1 = cv2.Sobel(first,cv2.CV_64F,0,3,ksize=5)
        
        x2 = cv2.Sobel(second,cv2.CV_64F,1,0)
        y2 = cv2.Sobel(second,cv2.CV_64F,0,1)
        xx2 = cv2.Sobel(second,cv2.CV_64F,2,0)
        yy2 = cv2.Sobel(second,cv2.CV_64F,0,2)
        xy2 = cv2.Sobel(second,cv2.CV_64F,1,1)
        xxx2 = cv2.Sobel(second,cv2.CV_64F,3,0,ksize=5)
        xxy2 = cv2.Sobel(second,cv2.CV_64F,2,1,ksize=5)
        xyy2 = cv2.Sobel(second,cv2.CV_64F,1,2,ksize=5)
        yyy2 = cv2.Sobel(second,cv2.CV_64F,0,3,ksize=5)
        
        u1 = first[vessel_ind][np.newaxis,:]
        ux1 = x1[vessel_ind][np.newaxis,:] # Derivatives that correspond to coordinate of vessels
        uy1 = y1[vessel_ind][np.newaxis,:]
        uxx1 = xx1[vessel_ind][np.newaxis,:]
        uyy1 = yy1[vessel_ind][np.newaxis,:]
        uxy1 = xy1[vessel_ind][np.newaxis,:]
        uxxx1 = xxx1[vessel_ind][np.newaxis,:]
        uxxy1 = xxy1[vessel_ind][np.newaxis,:]
        uxyy1 = xyy1[vessel_ind][np.newaxis,:]
        uyyy1 = yyy1[vessel_ind][np.newaxis,:]
        
        u2 = second[vessel_ind][np.newaxis,:]
        ux2 = x2[vessel_ind][np.newaxis,:]
        uy2 = y2[vessel_ind][np.newaxis,:]
        uxx2 = xx2[vessel_ind][np.newaxis,:]
        uyy2 = yy2[vessel_ind][np.newaxis,:]
        uxy2 = xy2[vessel_ind][np.newaxis,:]
        uxxx2 = xxx2[vessel_ind][np.newaxis,:]
        uxxy2 = xxy2[vessel_ind][np.newaxis,:]
        uxyy2 = xyy2[vessel_ind][np.newaxis,:]
        uyyy2 = yyy2[vessel_ind][np.newaxis,:]

        vessel_deriv_mat = np.concatenate((u1,ux1,uy1,uxx1,uyy1,uxy1,uxxx1,uxxy1,uxyy1,uyyy1,
                                           u2,ux2,uy2,uxx2,uyy2,uxy2,uxxx2,uxxy2,uxyy2,uyyy2),axis=0)
        
        u1 = first[non_vessel_ind][np.newaxis,:]
        ux1 = x1[non_vessel_ind][np.newaxis,:] # Derivatives that correspond to coordinate of vessels
        uy1 = y1[non_vessel_ind][np.newaxis,:]
        uxx1 = xx1[non_vessel_ind][np.newaxis,:]
        uyy1 = yy1[non_vessel_ind][np.newaxis,:]
        uxy1 = xy1[non_vessel_ind][np.newaxis,:]
        uxxx1 = xxx1[non_vessel_ind][np.newaxis,:]
        uxxy1 = xxy1[non_vessel_ind][np.newaxis,:]
        uxyy1 = xyy1[non_vessel_ind][np.newaxis,:]
        uyyy1 = yyy1[non_vessel_ind][np.newaxis,:]
        
        u2 = second[non_vessel_ind][np.newaxis,:]
        ux2 = x2[non_vessel_ind][np.newaxis,:]
        uy2 = y2[non_vessel_ind][np.newaxis,:]
        uxx2 = xx2[non_vessel_ind][np.newaxis,:]
        uyy2 = yy2[non_vessel_ind][np.newaxis,:]
        uxy2 = xy2[non_vessel_ind][np.newaxis,:]
        uxxx2 = xxx2[non_vessel_ind][np.newaxis,:]
        uxxy2 = xxy2[non_vessel_ind][np.newaxis,:]
        uxyy2 = xyy2[non_vessel_ind][np.newaxis,:]
        uyyy2 = yyy2[non_vessel_ind][np.newaxis,:]
        
        non_vessel_deriv_mat = np.concatenate((u1,ux1,uy1,uxx1,uyy1,uxy1,uxxx1,uxxy1,uxyy1,uyyy1,
                                               u2,ux2,uy2,uxx2,uyy2,uxy2,uxxx2,uxxy2,uxyy2,uyyy2),axis=0)
        return vessel_deriv_mat,non_vessel_deriv_mat
        
    def featureScale(self,feature_mat):
        feature_mean = feature_mat.mean(axis=0)
        feature_std = np.std(feature_mat,axis=0)
        feature_mean = np.repeat(feature_mean[np.newaxis,:],feature_mat.shape[0],axis=0)
        feature_std = np.repeat(feature_std[np.newaxis,:],feature_mat.shape[0],axis=0)
        scaled = (feature_mat-feature_mean)/feature_std
        return scaled
        
def getScaledImg(img,scale):
    sigma = np.sqrt(scale)
    size = int(np.ceil(sigma)*10+1)
    img = img.astype(np.float64)
    scaled_img = cv2.GaussianBlur(img,(size,size),sigma)
    return scaled_img