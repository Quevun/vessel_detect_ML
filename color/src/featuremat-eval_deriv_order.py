# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 09:46:31 2016

@author: Quek Yu Yang
"""
import numpy as np
import cv2

class FeatureMatMaker(object):
    def __init__(self,img,scales):
        assert np.size(img,2) == 3
        assert img.dtype == 'uint8'
        self.img = img
        self.scales = scales
        
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
            
            x3 = cv2.Sobel(third,cv2.CV_64F,1,0)[:,:,np.newaxis]
            y3 = cv2.Sobel(third,cv2.CV_64F,0,1)[:,:,np.newaxis]
            xx3 = cv2.Sobel(third,cv2.CV_64F,2,0)[:,:,np.newaxis]
            yy3 = cv2.Sobel(third,cv2.CV_64F,0,2)[:,:,np.newaxis]
            xy3 = cv2.Sobel(third,cv2.CV_64F,1,1)[:,:,np.newaxis]
            xxx3 = cv2.Sobel(third,cv2.CV_64F,3,0,ksize=5)[:,:,np.newaxis]
            xxy3 = cv2.Sobel(third,cv2.CV_64F,2,1,ksize=5)[:,:,np.newaxis]
            xyy3 = cv2.Sobel(third,cv2.CV_64F,1,2,ksize=5)[:,:,np.newaxis]
            yyy3 = cv2.Sobel(third,cv2.CV_64F,0,3,ksize=5)[:,:,np.newaxis]
            
            first = first[:,:,np.newaxis]
            second = second[:,:,np.newaxis]
            third = third[:,:,np.newaxis]
            
            if self.order == 1:
                feature_cube[:,:,nf*i+1:nf*i+nf+1] = np.concatenate((first,x1,y1,
                                                                     second,x2,y2,
                                                                     third,x3,y3),
                                                                     axis=2)
            elif self.order == 2:
                feature_cube[:,:,nf*i+1:nf*i+nf+1] = np.concatenate((first,x1,y1,xx1,yy1,xy1,
                                                                     second,x2,y2,xx2,yy2,xy2,
                                                                     third,x3,y3,xx3,yy3,xy3),
                                                                     axis=2)
            elif self.order == 3:
                feature_cube[:,:,nf*i+1:nf*i+nf+1] = np.concatenate((first,x1,y1,xx1,yy1,xy1,xxx1,xxy1,xyy1,yyy1,
                                                                     second,x2,y2,xx2,yy2,xy2,xxx2,xxy2,xyy2,yyy2,
                                                                     third,x3,y3,xx3,yy3,xy3,xxx3,xxy3,xyy3,yyy3),
                                                                     axis=2)
            
        feature_mat = np.reshape(feature_cube,(shape[0]*shape[1],shape[2]),'F')
        feature_mat = self.featureScale(feature_mat)
        feature_mat[:,0] = 1  # First element of each feature vector is 1
        return feature_mat
            
    def getTrainMat(self,vessel_ind,order):
        
        #######################
        #   Initialization
        self.vessel_ind = vessel_ind
        assert (order > 0) and (order <=3)
        self.order = order
        if order == 1:
            self.num_features = 9
        elif order == 2:
            self.num_features = 18
        elif order == 3:
            self.num_features = 30
        self.vessel_sample_size = self.vessel_ind[0].size
        num_scales = len(self.scales)
        vessel_v = [np.zeros((num_scales,self.num_features)) for _ in xrange(self.vessel_sample_size)]
        
        non_vessel_ind = self.getRandInd()
        non_vessel_sample_size = non_vessel_ind[0].size
        non_vessel_v = [np.zeros((num_scales,self.num_features)) for _ in xrange(non_vessel_sample_size)]
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
        first = scaled[:,:,0]   #first,second and third slice in axis 2
        second = scaled[:,:,1]
        third = scaled[:,:,2]
        
        x1 = cv2.Sobel(first,cv2.CV_64F,1,0)
        y1 = cv2.Sobel(first,cv2.CV_64F,0,1)
        x2 = cv2.Sobel(second,cv2.CV_64F,1,0)
        y2 = cv2.Sobel(second,cv2.CV_64F,0,1)
        x3 = cv2.Sobel(third,cv2.CV_64F,1,0)
        y3 = cv2.Sobel(third,cv2.CV_64F,0,1)
        u1 = first[vessel_ind][np.newaxis,:]
        ux1 = x1[vessel_ind][np.newaxis,:] # Derivatives that correspond to coordinate of vessels
        uy1 = y1[vessel_ind][np.newaxis,:]
        u2 = second[vessel_ind][np.newaxis,:]
        ux2 = x2[vessel_ind][np.newaxis,:]
        uy2 = y2[vessel_ind][np.newaxis,:]
        u3 = third[vessel_ind][np.newaxis,:]
        ux3 = x3[vessel_ind][np.newaxis,:]
        uy3 = y3[vessel_ind][np.newaxis,:]
        t1 = first[non_vessel_ind][np.newaxis,:]
        tx1 = x1[non_vessel_ind][np.newaxis,:] # Derivatives that correspond to coordinate of vessels
        ty1 = y1[non_vessel_ind][np.newaxis,:]
        t2 = second[non_vessel_ind][np.newaxis,:]
        tx2 = x2[non_vessel_ind][np.newaxis,:]
        ty2 = y2[non_vessel_ind][np.newaxis,:]
        t3 = third[non_vessel_ind][np.newaxis,:]
        tx3 = x3[non_vessel_ind][np.newaxis,:]
        ty3 = y3[non_vessel_ind][np.newaxis,:]
        
        if self.order >= 2:
            xx1 = cv2.Sobel(first,cv2.CV_64F,2,0)
            yy1 = cv2.Sobel(first,cv2.CV_64F,0,2)
            xy1 = cv2.Sobel(first,cv2.CV_64F,1,1)
            xx2 = cv2.Sobel(second,cv2.CV_64F,2,0)
            yy2 = cv2.Sobel(second,cv2.CV_64F,0,2)
            xy2 = cv2.Sobel(second,cv2.CV_64F,1,1)
            xx3 = cv2.Sobel(third,cv2.CV_64F,2,0)
            yy3 = cv2.Sobel(third,cv2.CV_64F,0,2)
            xy3 = cv2.Sobel(third,cv2.CV_64F,1,1)
            uxx1 = xx1[vessel_ind][np.newaxis,:]
            uyy1 = yy1[vessel_ind][np.newaxis,:]
            uxy1 = xy1[vessel_ind][np.newaxis,:]
            uxx2 = xx2[vessel_ind][np.newaxis,:]
            uyy2 = yy2[vessel_ind][np.newaxis,:]
            uxy2 = xy2[vessel_ind][np.newaxis,:]
            uxx3 = xx3[vessel_ind][np.newaxis,:]
            uyy3 = yy3[vessel_ind][np.newaxis,:]
            uxy3 = xy3[vessel_ind][np.newaxis,:]
            txx1 = xx1[non_vessel_ind][np.newaxis,:]
            tyy1 = yy1[non_vessel_ind][np.newaxis,:]
            txy1 = xy1[non_vessel_ind][np.newaxis,:]
            txx2 = xx2[non_vessel_ind][np.newaxis,:]
            tyy2 = yy2[non_vessel_ind][np.newaxis,:]
            txy2 = xy2[non_vessel_ind][np.newaxis,:]
            txx3 = xx3[non_vessel_ind][np.newaxis,:]
            tyy3 = yy3[non_vessel_ind][np.newaxis,:]
            txy3 = xy3[non_vessel_ind][np.newaxis,:]
            
        if self.order == 3:
            xxx1 = cv2.Sobel(first,cv2.CV_64F,3,0,ksize=5)
            xxy1 = cv2.Sobel(first,cv2.CV_64F,2,1,ksize=5)
            xyy1 = cv2.Sobel(first,cv2.CV_64F,1,2,ksize=5)
            yyy1 = cv2.Sobel(first,cv2.CV_64F,0,3,ksize=5)
            xxx2 = cv2.Sobel(second,cv2.CV_64F,3,0,ksize=5)
            xxy2 = cv2.Sobel(second,cv2.CV_64F,2,1,ksize=5)
            xyy2 = cv2.Sobel(second,cv2.CV_64F,1,2,ksize=5)
            yyy2 = cv2.Sobel(second,cv2.CV_64F,0,3,ksize=5)
            xxx3 = cv2.Sobel(third,cv2.CV_64F,3,0,ksize=5)
            xxy3 = cv2.Sobel(third,cv2.CV_64F,2,1,ksize=5)
            xyy3 = cv2.Sobel(third,cv2.CV_64F,1,2,ksize=5)
            yyy3 = cv2.Sobel(third,cv2.CV_64F,0,3,ksize=5) 
            uxxx1 = xxx1[vessel_ind][np.newaxis,:]
            uxxy1 = xxy1[vessel_ind][np.newaxis,:]
            uxyy1 = xyy1[vessel_ind][np.newaxis,:]
            uyyy1 = yyy1[vessel_ind][np.newaxis,:] 
            uxxx2 = xxx2[vessel_ind][np.newaxis,:]
            uxxy2 = xxy2[vessel_ind][np.newaxis,:]
            uxyy2 = xyy2[vessel_ind][np.newaxis,:]
            uyyy2 = yyy2[vessel_ind][np.newaxis,:]
            uxxx3 = xxx3[vessel_ind][np.newaxis,:]
            uxxy3 = xxy3[vessel_ind][np.newaxis,:]
            uxyy3 = xyy3[vessel_ind][np.newaxis,:]
            uyyy3 = yyy3[vessel_ind][np.newaxis,:]
            txxx1 = xxx1[non_vessel_ind][np.newaxis,:]
            txxy1 = xxy1[non_vessel_ind][np.newaxis,:]
            txyy1 = xyy1[non_vessel_ind][np.newaxis,:]
            tyyy1 = yyy1[non_vessel_ind][np.newaxis,:]
            txxx2 = xxx2[non_vessel_ind][np.newaxis,:]
            txxy2 = xxy2[non_vessel_ind][np.newaxis,:]
            txyy2 = xyy2[non_vessel_ind][np.newaxis,:]
            tyyy2 = yyy2[non_vessel_ind][np.newaxis,:]
            txxx3 = xxx3[non_vessel_ind][np.newaxis,:]
            txxy3 = xxy3[non_vessel_ind][np.newaxis,:]
            txyy3 = xyy3[non_vessel_ind][np.newaxis,:]
            tyyy3 = yyy3[non_vessel_ind][np.newaxis,:]
            
        if self.order == 1:
            vessel_deriv_mat = np.concatenate((u1,ux1,uy1,
                                               u2,ux2,uy2,
                                               u3,ux3,uy3),axis=0)
            non_vessel_deriv_mat = np.concatenate((t1,tx1,ty1,
                                                   t2,tx2,ty2,
                                                   t3,tx3,ty3),axis=0)
                                                   
        elif self.order == 2:
            self.num_features = 18
            vessel_deriv_mat = np.concatenate((u1,ux1,uy1,uxx1,uyy1,uxy1,
                                               u2,ux2,uy2,uxx2,uyy2,uxy2,
                                               u3,ux3,uy3,uxx3,uyy3,uxy3),axis=0)
            non_vessel_deriv_mat = np.concatenate((t1,tx1,ty1,txx1,tyy1,txy1,
                                                   t2,tx2,ty2,txx2,tyy2,txy2,
                                                   t3,tx3,ty3,txx3,tyy3,txy3),axis=0)
                                               
        elif self.order == 3:
            self.num_features = 30
            vessel_deriv_mat = np.concatenate((u1,ux1,uy1,uxx1,uyy1,uxy1,uxxx1,uxxy1,uxyy1,uyyy1,
                                               u2,ux2,uy2,uxx2,uyy2,uxy2,uxxx2,uxxy2,uxyy2,uyyy2,
                                               u3,ux3,uy3,uxx3,uyy3,uxy3,uxxx3,uxxy3,uxyy3,uyyy3),axis=0)
            non_vessel_deriv_mat = np.concatenate((t1,tx1,ty1,txx1,tyy1,txy1,txxx1,txxy1,txyy1,tyyy1,
                                                   t2,tx2,ty2,txx2,tyy2,txy2,txxx2,txxy2,txyy2,tyyy2,
                                                   t3,tx3,ty3,txx3,tyy3,txy3,txxx3,txxy3,txyy3,tyyy3),axis=0)
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