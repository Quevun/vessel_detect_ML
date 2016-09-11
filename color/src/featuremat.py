# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 09:46:31 2016

@author: Quek Yu Yang
"""
import numpy as np
import cv2

class FeatureMatMaker(object):
    def __init__(self,img,vessel_ind,scales):
        assert np.size(img,2) == 3
        assert img.dtype == 'uint8'
        self.img = img
        self.vessel_ind = vessel_ind
        self.scales = scales
        self.vessel_sample_size = np.size(vessel_ind[0])
        
    def getMat(self):
        shape = (np.size(self.img,0),np.size(self.img,1),len(self.scales)*15+1)
        feature_cube = np.zeros(shape)
        feature_cube[:,:,0] = 1  # First element of each feature vector is 1
        for i in range(len(self.scales)):
            scaled = getScaledImg(self.img,self.scales[i])
            first = scaled[:,:,0]   #first,second and third slice in axis 2
            second = scaled[:,:,1]
            third = scaled[:,:,2]
            
            x1 = cv2.Sobel(first,cv2.CV_64F,1,0)[:,:,np.newaxis]
            y1 = cv2.Sobel(first,cv2.CV_64F,0,1)[:,:,np.newaxis]
            xx1 = cv2.Sobel(first,cv2.CV_64F,2,0)[:,:,np.newaxis]
            yy1 = cv2.Sobel(first,cv2.CV_64F,0,2)[:,:,np.newaxis]
            xy1 = cv2.Sobel(first,cv2.CV_64F,1,1)[:,:,np.newaxis]
            x2 = cv2.Sobel(second,cv2.CV_64F,1,0)[:,:,np.newaxis]
            y2 = cv2.Sobel(second,cv2.CV_64F,0,1)[:,:,np.newaxis]
            xx2 = cv2.Sobel(second,cv2.CV_64F,2,0)[:,:,np.newaxis]
            yy2 = cv2.Sobel(second,cv2.CV_64F,0,2)[:,:,np.newaxis]
            xy2 = cv2.Sobel(second,cv2.CV_64F,1,1)[:,:,np.newaxis]
            x3 = cv2.Sobel(third,cv2.CV_64F,1,0)[:,:,np.newaxis]
            y3 = cv2.Sobel(third,cv2.CV_64F,0,1)[:,:,np.newaxis]
            xx3 = cv2.Sobel(third,cv2.CV_64F,2,0)[:,:,np.newaxis]
            yy3 = cv2.Sobel(third,cv2.CV_64F,0,2)[:,:,np.newaxis]
            xy3 = cv2.Sobel(third,cv2.CV_64F,1,1)[:,:,np.newaxis]
            
            feature_cube[:,:,15*i+1:15*i+15+1] = np.concatenate((x1,y1,xx1,yy1,xy1,
                                                                 x2,y2,xx2,yy2,xy2,
                                                                 x3,y3,xx3,yy3,xy3),axis=2)
        feature_mat = np.reshape(feature_cube,(shape[0]*shape[1],shape[2]),'F')
        return feature_mat
        
    def getTrainMat(self,is_vessel = True):
        if is_vessel:   # Extract vessel features
            num_scales = len(self.scales)
            v = [np.zeros((num_scales,15)) for _ in xrange(self.vessel_sample_size)]
            
            for i in range(num_scales): 
                scaled = self.getScaledImg(self.img,self.scales[i])
                deriv_mat = self.getDerivMat(scaled,self.vessel_ind)
                for j in range(self.vessel_sample_size):
                    v[j][i,:] = deriv_mat[:,j]
            feature_mat = np.zeros((self.vessel_sample_size,15*num_scales))
            for i in range(len(v)):
                feature_mat[i,:] = v[i].flatten()
            return feature_mat
            
        elif not is_vessel: #Extract random non-vessel features
            y = self.vessel_ind[0][np.newaxis,:]
            x = self.vessel_ind[1][np.newaxis,:]
            vessel_ind_struc = np.concatenate((y,x),axis=0).flatten('F')
            vessel_ind_struc = vessel_ind_struc.view([('y',np.int64),
                                                      ('x',np.int64)]) # structured array with (y,x) elements
            y = np.random.randint(self.img.shape[0],size=self.vessel_sample_size)[np.newaxis,:]
            x = np.random.randint(self.img.shape[1],size=self.vessel_sample_size)[np.newaxis,:]
            non_vessel_ind_struc = np.concatenate((y,x),axis=0).flatten('F')
            non_vessel_ind_struc = non_vessel_ind_struc.astype(np.int64)
            non_vessel_ind_struc = non_vessel_ind_struc.view([('y',np.int64),
                                                              ('x',np.int64)])
            non_vessel_ind_struc = np.unique(non_vessel_ind_struc)
        
            intersects = np.intersect1d(vessel_ind_struc,non_vessel_ind_struc,True)
            for intersect in intersects:
                non_vessel_ind_struc = np.delete(non_vessel_ind_struc, np.where(non_vessel_ind_struc==intersect))
            non_vessel_ind = (non_vessel_ind_struc['y'],non_vessel_ind_struc['x']) 
            
            num_scales = len(self.scales)
            non_vessel_sample_size = non_vessel_ind[0].size
            v = [np.zeros((num_scales,15)) for _ in xrange(non_vessel_sample_size)]
            
            for i in range(num_scales): 
                scaled = self.getScaledImg(self.img,self.scales[i])
                deriv_mat = self.getDerivMat(scaled,non_vessel_ind)
                for j in range(non_vessel_sample_size):
                    v[j][i,:] = deriv_mat[:,j]
            feature_mat = np.zeros((non_vessel_sample_size,15*num_scales))
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
        
def getScaledImg(img,scale):
    sigma = np.sqrt(scale)
    size = int(np.ceil(sigma)*10+1)
    img = img.astype(np.float64)
    scaled_img = cv2.GaussianBlur(img,(size,size),sigma)
    return scaled_img