# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:15:11 2016

@author: queky
"""

import cv2
import numpy as np

############################################################################
def hessEig(sobelxx,sobelxy,sobelyy):
    eigval_array = np.zeros((np.size(sobelxx,0),np.size(sobelxx,1),2))   # initialization
    for i in range(np.size(sobelxx,0)):   # row index
        for j in range(np.size(sobelxx,1)):   # column index
            mat = [[sobelxx[i,j],sobelxy[i,j]],[sobelxy[i,j],sobelyy[i,j]]]   #hessian matrix
            [eigval,eigvec] = np.linalg.eig(mat)
            eigval_array[i,j,:] = eigval
    return eigval_array
    
###########################################################################
filename = 'yamaki4'
img = cv2.imread('../data/IR/'+filename+'.bmp',cv2.IMREAD_GRAYSCALE)
small = cv2.pyrDown(img)
small = cv2.GaussianBlur(small,(21,21),2)

sobelxx = cv2.Sobel(small,cv2.CV_64F,2,0,ksize=9)
sobelyy = cv2.Sobel(small,cv2.CV_64F,0,2,ksize=9)
sobelxy = cv2.Sobel(small,cv2.CV_64F,1,1,ksize=9)

maj_thres = 2000
#min_thres = 1
eigval_array = hessEig(sobelxx,sobelxy,sobelyy)
major = np.amax(eigval_array,2) > maj_thres
#minor = abs(np.amin(eigval_array,2)) < min_thres
#bin = np.logical_and(major,minor)
#bin = bin.astype(np.uint8) * 255

#upscale = cv2.pyrUp(major.astype(np.uint8)*255)

major = major.astype(np.uint8)*255
cv2.imshow('stuff',major)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('../data/IR/small/'+filename+'.jpg',small)
np.save('../data/eigen/'+filename,(major==255).astype(np.uint8)*255)