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
        self.num_features = 18
        
    def getMat(self):
        nf = self.num_features   # Short for number of features
        shape = (np.size(self.img,0),np.size(self.img,1),len(self.scales)*nf+1)
        feature_cube = np.zeros(shape)
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
            
            first = first[:,:,np.newaxis]
            second = second[:,:,np.newaxis]
            third = third[:,:,np.newaxis]
            feature_cube[:,:,nf*i+1:nf*i+nf+1] = np.concatenate((first,x1,y1,xx1,yy1,xy1,
                                                                 second,x2,y2,xx2,yy2,xy2,
                                                                 third,x3,y3,xx3,yy3,xy3),
                                                                 axis=2)
            
        feature_mat = np.reshape(feature_cube,(shape[0]*shape[1],shape[2]),'F')
        feature_mat = self.featureScale(feature_mat)
        feature_mat[:,0] = 1  # First element of each feature vector is 1
        return feature_mat
            
    def getTrainMat(self,vessel_ind):
        
        #######################
        #   Initialization
        self.vessel_ind = vessel_ind
        self.vessel_sample_size = self.vessel_ind[0].size
        num_scales = len(self.scales)
        rot_angles = np.arange(10,180,10)
        vessel_v = [np.zeros((num_scales,self.num_features)) for _ in xrange(self.vessel_sample_size*len(rot_angles))]
        
        non_vessel_ind = self.getRandInd()
        non_vessel_sample_size = non_vessel_ind[0].size
        non_vessel_v = [np.zeros((num_scales,self.num_features)) for _ in xrange(non_vessel_sample_size*len(rot_angles))]
        
        #######################        
        
        for i in range(num_scales): 
            scaled = getScaledImg(self.img,self.scales[i])
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
        sample_size = np.around(self.img.shape[0]*self.img.shape[1]/16)
        y = np.random.randint(self.img.shape[0],size=sample_size)[np.newaxis,:]
        x = np.random.randint(self.img.shape[1],size=sample_size)[np.newaxis,:]
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
        
        u1 = first[vessel_ind][np.newaxis,:]
        ux1 = x1[vessel_ind][np.newaxis,:] # Derivatives that correspond to coordinate of vessels
        uy1 = y1[vessel_ind][np.newaxis,:]
        uxx1 = xx1[vessel_ind][np.newaxis,:]
        uyy1 = yy1[vessel_ind][np.newaxis,:]
        uxy1 = xy1[vessel_ind][np.newaxis,:]
        
        u2 = second[vessel_ind][np.newaxis,:]
        ux2 = x2[vessel_ind][np.newaxis,:]
        uy2 = y2[vessel_ind][np.newaxis,:]
        uxx2 = xx2[vessel_ind][np.newaxis,:]
        uyy2 = yy2[vessel_ind][np.newaxis,:]
        uxy2 = xy2[vessel_ind][np.newaxis,:]
        
        u3 = third[vessel_ind][np.newaxis,:]
        ux3 = x3[vessel_ind][np.newaxis,:]
        uy3 = y3[vessel_ind][np.newaxis,:]
        uxx3 = xx3[vessel_ind][np.newaxis,:]
        uyy3 = yy3[vessel_ind][np.newaxis,:]
        uxy3 = xy3[vessel_ind][np.newaxis,:]
                                     
        vessel_deriv_mat = np.concatenate((u1,ux1,uy1,uxx1,uyy1,uxy1,
                                           u2,ux2,uy2,uxx2,uyy2,uxy2,
                                           u3,ux3,uy3,uxx3,uyy3,uxy3,),axis=0)
                                           
        u1 = first[non_vessel_ind][np.newaxis,:]
        ux1 = x1[non_vessel_ind][np.newaxis,:] # Derivatives that correspond to coordinate of vessels
        uy1 = y1[non_vessel_ind][np.newaxis,:]
        uxx1 = xx1[non_vessel_ind][np.newaxis,:]
        uyy1 = yy1[non_vessel_ind][np.newaxis,:]
        uxy1 = xy1[non_vessel_ind][np.newaxis,:]
        
        u2 = second[non_vessel_ind][np.newaxis,:]
        ux2 = x2[non_vessel_ind][np.newaxis,:]
        uy2 = y2[non_vessel_ind][np.newaxis,:]
        uxx2 = xx2[non_vessel_ind][np.newaxis,:]
        uyy2 = yy2[non_vessel_ind][np.newaxis,:]
        uxy2 = xy2[non_vessel_ind][np.newaxis,:]
        
        u3 = third[non_vessel_ind][np.newaxis,:]
        ux3 = x3[non_vessel_ind][np.newaxis,:]
        uy3 = y3[non_vessel_ind][np.newaxis,:]
        uxx3 = xx3[non_vessel_ind][np.newaxis,:]
        uyy3 = yy3[non_vessel_ind][np.newaxis,:]
        uxy3 = xy3[non_vessel_ind][np.newaxis,:]

        non_vessel_deriv_mat = np.concatenate((u1,ux1,uy1,uxx1,uyy1,uxy1,
                                               u2,ux2,uy2,uxx2,uyy2,uxy2,
                                               u3,ux3,uy3,uxx3,uyy3,uxy3,),axis=0)
        return vessel_deriv_mat,non_vessel_deriv_mat
        
    def rotateImgAndInd(self,img,vessel_ind,non_vessel_ind,rot_angle):
        diag_len = round(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
        x_pad = round((diag_len - img.shape[1] + 5)/2)*2
        y_pad = round((diag_len - img.shape[0] + 5)/2)*2
        padded_img = np.zeros((img.shape[0]+y_pad,img.shape[1]+x_pad,img.shape[2]))
        x_mid = round(padded_img.shape[1]/2)
        y_mid = round(padded_img.shape[0]/2)
        
        padded_img[y_pad/2:-y_pad/2,x_pad/2:-x_pad/2,:] = img
        padded_img = padded_img.astype(np.uint8)
        
        rotated_imgs = []
        for angle in rot_angle:
            rot_mat = cv2.getRotationMatrix2D((x_mid,y_mid),angle,1)
            rotated_img = np.zeros((padded_img.shape))
            rotated_img[:,:,0] = cv2.warpAffine(padded_img[:,:,0],rot_mat
                                               ,(padded_img.shape[1],padded_img.shape[0]))
            rotated_img[:,:,1] = cv2.warpAffine(padded_img[:,:,1],rot_mat
                                               ,(padded_img.shape[1],padded_img.shape[0]))
            rotated_img[:,:,2] = cv2.warpAffine(padded_img[:,:,2],rot_mat
                                               ,(padded_img.shape[1],padded_img.shape[0]))
            rotated_img = rotated_img.astype(np.uint8)
            rotated_imgs.append(rotated_img)
            
        vessel_bin = np.zeros((self.img.shape[0],self.img.shape[1])).astype(np.uint8)
        vessel_bin[vessel_ind] = 255
        return rotated_imgs
        
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