# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 16:38:02 2016

@author: Quek Yu Yang
"""
import numpy as np
import cv2

def extractFeatures(img,scales):
    shape = (np.size(img,0),np.size(img,1),len(scales)*5+1)
    feature_cube = np.zeros(shape)
    feature_cube[:,:,0] = 1  # First element of each feature vector is 1
    for i in range(len(scales)):
        scaled = ScaledImg(img,scales[i])
        Ix = scaled.getDerivX()[:,:,np.newaxis]
        Iy = scaled.getDerivY()[:,:,np.newaxis]
        Ixx = scaled.getDerivXX()[:,:,np.newaxis]
        Iyy = scaled.getDerivYY()[:,:,np.newaxis]
        Ixy = scaled.getDerivXY()[:,:,np.newaxis]
        feature_cube[:,:,5*i+1:5*i+5+1] = np.concatenate((Ix,Iy,Ixx,Iyy,Ixy),axis=2)
    feature_mat = np.reshape(feature_cube,(shape[0]*shape[1],shape[2]),'F')
    return feature_mat

def makeFeatureMatrix(img,index,scales):
    sample_size = np.size(index[0])
    v = [np.zeros((len(scales),5)) for _ in xrange(sample_size)]
    
    for i in range(len(scales)): 
        scaled = ScaledImg(img,scales[i])
        Ix = scaled.getDerivX()
        Iy = scaled.getDerivY()
        Ixx = scaled.getDerivXX()
        Iyy = scaled.getDerivYY()
        Ixy = scaled.getDerivXY()
        
        ux = Ix[index][np.newaxis,:] # Derivatives that correspond to coordinate of vessels
        uy = Iy[index][np.newaxis,:]
        uxx = Ixx[index][np.newaxis,:]
        uyy = Iyy[index][np.newaxis,:]
        uxy = Ixy[index][np.newaxis,:]
    
        temp = np.concatenate((ux,uy,uxx,uyy,uxy),axis=0)
        for j in range(sample_size):
            v[j][i,:] = temp[:,j]
    feature_mat = np.zeros((sample_size,5*len(scales)))
    for i in range(len(v)):
        feature_mat[i,:] = v[i].flatten()
    return feature_mat
    
def getScaledImg(img,scale):
    sigma = np.sqrt(scale)
    size = int(np.ceil(sigma)*10+1)
    img = img.astype(np.float64)
    scaled_img = cv2.GaussianBlur(img,(size,size),sigma)
    #scaled_img = cv2.medianBlur(img,size)    
    return scaled_img
    
class Img(object):
    def getDerivX(self):   # Might want to implement these at the cuboid level for efficiency
        self.derivX = np.zeros(np.shape(self.img))
        for i in range(np.size(self.img,1)):
            self.derivX[:,i] = self.img[:,(i+1)%np.size(self.img,1)] - self.img[:,i-1]
        return self.derivX
        
    def getDerivY(self):
        self.derivY = np.zeros(np.shape(self.img))
        for i in range(np.size(self.img,0)):
            self.derivY[i,:] = self.img[(i+1)%np.size(self.img,0),:] - self.img[i-1,:]
        return self.derivY
        
    def getDerivXX(self):
        self.derivXX = np.zeros(np.shape(self.img))
        for i in range(np.size(self.img,1)):
            self.derivXX[:,i]=self.derivX[:,(i+1)%np.size(self.img,1)]-self.derivX[:,i-1]
        return self.derivXX
        
    def getDerivYY(self):
        self.derivYY = np.zeros(np.shape(self.img))
        for i in range(np.size(self.img,0)):
            self.derivYY[i,:]=self.derivY[(i+1)%np.size(self.img,0),:]-self.derivY[i-1,:]
        return self.derivYY
        
    def getDerivXY(self):
        self.derivXY = np.zeros(np.shape(self.img))
        for i in range(np.size(self.img,0)):
            self.derivXY[i,:]=self.derivX[(i+1)%np.size(self.img,0),:]-self.derivX[i-1,:]
        return self.derivXY
        
class ScaledImg(Img):
    def __init__(self,img,scale):
        self.scale = scale
        self.img = getScaledImg(img,scale) # floating point image
        self.sobelx = None
        self.sobely = None
        self.sobelxx = None
        self.sobelyy = None
        self.sobelxy = None
        
    def getImg(self):
        return self.img
        
    def getScale(self):
        return self.scale
        
    def getSobelx(self):
        if self.sobelx is None:
            self.sobelx = cv2.Sobel(self.img,cv2.CV_64F,1,0)
            return self.sobelx
        else:
            return self.sobelx
            
    def getSobely(self):
        if self.sobely is None:
            self.sobely = cv2.Sobel(self.img,cv2.CV_64F,0,1)
            return self.sobely
        else:
            return self.sobely
        
    def getSobelxx(self):
        if self.sobelxx is None:
            self.sobelxx = cv2.Sobel(self.img,cv2.CV_64F,2,0)#,ksize=scale + scale % 2 - 1)
            return self.sobelxx
        else:
            return self.sobelxx
            
    def getSobelyy(self):
        if self.sobelyy is None:
            self.sobelyy = cv2.Sobel(self.img,cv2.CV_64F,0,2)#,ksize=scale + scale % 2 - 1)
            return self.sobelyy
        else:
            return self.sobelyy
            
    def getSobelxy(self):
        if self.sobelxy is None:
            self.sobelxy = cv2.Sobel(self.img,cv2.CV_64F,1,1)#,ksize=scale + scale % 2 - 1)
            return self.sobelxy
        else:
            return self.sobelxy