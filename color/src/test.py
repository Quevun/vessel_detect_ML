# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 11:11:37 2016

@author: keisoku
"""
import cv2
import numpy as np
import featuremat

"""#test FeatureMat
vessel_bin = np.load('../data/vessels/tanaka1.npy')
vessel_index = np.nonzero(vessel_bin)
scales = np.arange(3,200,5)
img = cv2.imread('../data/color/tanaka1.bmp')

###############################################
#   vessels

color_feature_mat = featuremat.FeatureMatMaker(img,vessel_index,scales).getMat()
gray_feature_mat1 = makeTrainSet2.makeFeatureMatrix(img[:,:,0],vessel_index,scales)
gray_feature_mat2 = makeTrainSet2.makeFeatureMatrix(img[:,:,1],vessel_index,scales)
gray_feature_mat3 = makeTrainSet2.makeFeatureMatrix(img[:,:,2],vessel_index,scales)

for i in range(len(scales)):
    print np.array_equal(color_feature_mat[:,i*15:i*15+5],gray_feature_mat1[:,i*5:i*5+5])
    print np.array_equal(color_feature_mat[:,i*15+5:i*15+5+5],gray_feature_mat2[:,i*5:i*5+5])
    print np.array_equal(color_feature_mat[:,i*15+10:i*15+10+5],gray_feature_mat3[:,i*5:i*5+5])
##############################################


##############################################
#   non-vessels

color_feature_mat,non_vessel_ind = featuremat.FeatureMatMaker(img,vessel_index,scales).getMat(False)
gray_feature_mat1 = makeTrainSet2.makeFeatureMatrix(img[:,:,0],non_vessel_ind,scales)
gray_feature_mat2 = makeTrainSet2.makeFeatureMatrix(img[:,:,1],non_vessel_ind,scales)
gray_feature_mat3 = makeTrainSet2.makeFeatureMatrix(img[:,:,2],non_vessel_ind,scales)

for i in range(len(scales)):
    print np.array_equal(color_feature_mat[:,i*15:i*15+5],gray_feature_mat1[:,i*5:i*5+5])
    print np.array_equal(color_feature_mat[:,i*15+5:i*15+5+5],gray_feature_mat2[:,i*5:i*5+5])
    print np.array_equal(color_feature_mat[:,i*15+10:i*15+10+5],gray_feature_mat3[:,i*5:i*5+5])
#############################################
"""


"""# test FeatureMat.getTraintMat
scales = np.arange(3,100,5)
img = cv2.imread('../data/color/tanaka2.bmp')
vessel_bin = np.load('../data/vessels/tanaka2.npy')
vessel_ind = np.nonzero(vessel_bin)
featuremat_maker = featuremat.FeatureMatMaker(img,scales)
vessel_feature,non_vessel_feature = featuremat_maker.getTrainMat(vessel_ind)
"""

"""# gradient direction image
img = cv2.imread('../data/IR/quek1.bmp',0)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
grad = np.arctan(sobely/sobelx)
"""

"""# ridge orientation
img = cv2.imread('../data/IR/quek1.bmp',0)
img = img.astype(np.float64)
img = cv2.GaussianBlur(img,(15,15),3)
Lxx = cv2.Sobel(img,cv2.CV_64F,2,0)
Lyy = cv2.Sobel(img,cv2.CV_64F,0,2)
Lxy = cv2.Sobel(img,cv2.CV_64F,1,1)
temp = (Lxx - Lyy)/np.sqrt((Lxx-Lyy)**2 + 4*Lxy**2)
sin_beta = np.sign(Lxy) * np.sqrt((1-temp)/2)
cos_beta = np.sqrt((1+temp)/2)
theta_p = np.arctan(-cos_beta/sin_beta)
theta_q = np.arctan(sin_beta/cos_beta)

theta_p = np.nan_to_num(theta_p)
theta_q =np.nan_to_num(theta_q)

theta_p = theta_p - np.amin(theta_p)
theta_p = (theta_p/np.amax(theta_p)*255).astype(np.uint8)
theta_q = theta_q - np.amin(theta_q)
theta_q = (theta_q/np.amax(theta_q)*255).astype(np.uint8)
"""

"""# test to see how image rotation works at small scale
# results: rotating an image smoothes it somewhat, avoid multiple rotations
img = (np.random.rand(3,3,3) * 255).astype(np.uint8)
rot_mat = cv2.getRotationMatrix2D((1,1),30,1)
img30 = cv2.warpAffine(img,rot_mat,(3,3),cv2.INTER_NEAREST)
img60 = cv2.warpAffine(img30,rot_mat,(3,3),cv2.INTER_NEAREST)
img90 = cv2.warpAffine(img60,rot_mat,(3,3),cv2.INTER_NEAREST)
img120 = cv2.warpAffine(img90,rot_mat,(3,3),cv2.INTER_NEAREST)
img150 = cv2.warpAffine(img120,rot_mat,(3,3),cv2.INTER_NEAREST)
img180 = cv2.warpAffine(img150,rot_mat,(3,3),cv2.INTER_NEAREST)

img = cv2.resize(img,(300,300),interpolation=cv2.INTER_NEAREST)
img30 = cv2.resize(img30,(300,300),interpolation=cv2.INTER_NEAREST)
img60 = cv2.resize(img60,(300,300),interpolation=cv2.INTER_NEAREST)
img90 = cv2.resize(img90,(300,300),interpolation=cv2.INTER_NEAREST)
img120 = cv2.resize(img120,(300,300),interpolation=cv2.INTER_NEAREST)
img150 = cv2.resize(img150,(300,300),interpolation=cv2.INTER_NEAREST)
img180 = cv2.resize(img180,(300,300),interpolation=cv2.INTER_NEAREST)
cv2.imwrite('../data/junk/img.jpg',img)
cv2.imwrite('../data/junk/img30.jpg',img30)
cv2.imwrite('../data/junk/img60.jpg',img60)
cv2.imwrite('../data/junk/img90.jpg',img90)
cv2.imwrite('../data/junk/img120.jpg',img120)
cv2.imwrite('../data/junk/img150.jpg',img150)
cv2.imwrite('../data/junk/img180.jpg',img180)
"""

#test featuremat.FeatureMatMaker.rotateImgAndInd
scales = np.arange(3,50,5)
rot_angles = np.arange(10,180,10)
img = cv2.imread('../data/color/quek1.bmp')
img = cv2.pyrDown(img)
feature_mat_maker = featuremat.FeatureMatMaker(img,scales)
rotated_imgs = feature_mat_maker.rotateImgAndInd(img,1,1,rot_angles)