# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 11:11:37 2016

@author: keisoku
"""
import cv2
import numpy as np
import featuremat
import sys
import time

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


"""# gradient direction image
img = cv2.imread('../data/IR/quek1.bmp',0)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
grad = np.arctan(sobely/sobelx)
"""

"""# ridge orientation
img = cv2.imread('../data/color/quek1.bmp')
img = img[:,:,2]
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

"""#test featuremat.FeatureMatMaker.rotateImgAndInd
scales = np.arange(3,50,5)
rot_angles = np.arange(10,180,10)
img = (np.random.rand(6,6,3)*10).astype(np.uint8)
print img[:,:,0]
print ''
print img[:,:,1]
print ''
print img[:,:,2]

img = cv2.imread('../data/color/quek1.bmp')
img = cv2.pyrDown(img)
vessel_bin = np.load('../data/vessels/red/quek1.npy')
vessel_ind = np.nonzero(vessel_bin)

#vessel_ind = (np.array([1,3,1,2,3,4,2,4]),np.array([1,1,2,2,3,3,4,4]))
feature_mat_maker = featuremat.FeatureMatMaker(img,scales)
vessel_feature_mat,non_vessel_feature_mat = feature_mat_maker.getTrainMat(vessel_ind)
dummy = (np.array([2,3]),np.array([2,3]))
bkmrk = 0
(rotated_img,
 rotated_vessel_ind,
 rotated_non_vessel_ind) = feature_mat_maker.rotateImgAndInd(img,vessel_ind,dummy,rot_angles)

for h in range(len(rotated_img)):
    
    img = featuremat.getScaledImg(rotated_img[h],3)
    vessel_ind = rotated_vessel_ind[h]
    feature_mat = np.zeros((vessel_ind[0].size,30))
    for i in range(3):
        foo = img[:,:,i]
        sobelx = cv2.Sobel(foo,cv2.CV_64F,1,0)
        sobely = cv2.Sobel(foo,cv2.CV_64F,0,1)
        sobelxx = cv2.Sobel(foo,cv2.CV_64F,2,0)
        sobelyy = cv2.Sobel(foo,cv2.CV_64F,0,2)
        sobelxy = cv2.Sobel(foo,cv2.CV_64F,1,1)
        sobelxxx = cv2.Sobel(foo,cv2.CV_64F,3,0,ksize=5)
        sobelxxy = cv2.Sobel(foo,cv2.CV_64F,2,1,ksize=5)
        sobelxyy = cv2.Sobel(foo,cv2.CV_64F,1,2,ksize=5)
        sobelyyy = cv2.Sobel(foo,cv2.CV_64F,0,3,ksize=5)
        
        vessel_mat = img[:,:,i][vessel_ind][np.newaxis,:]
        deriv_matx = sobelx[vessel_ind][np.newaxis,:]
        deriv_maty = sobely[vessel_ind][np.newaxis,:]
        deriv_matxx = sobelxx[vessel_ind][np.newaxis,:]
        deriv_matyy = sobelyy[vessel_ind][np.newaxis,:]
        deriv_matxy = sobelxy[vessel_ind][np.newaxis,:]
        deriv_matxxx = sobelxxx[vessel_ind][np.newaxis,:]
        deriv_matxxy = sobelxxy[vessel_ind][np.newaxis,:]
        deriv_matxyy = sobelxyy[vessel_ind][np.newaxis,:]
        deriv_matyyy = sobelyyy[vessel_ind][np.newaxis,:]
        
        sub_feature_mat = np.concatenate((vessel_mat,deriv_matx,deriv_maty,
                                          deriv_matxx,deriv_matyy,deriv_matxy,
                                          deriv_matxxx,deriv_matxxy,deriv_matxyy,deriv_matyyy),
                                          axis=0)
        feature_mat[:,i*10:i*10+10] = sub_feature_mat.T
    vessel_samples = vessel_ind[0].size
    print np.array_equal(vessel_feature_mat[bkmrk:bkmrk+vessel_samples,:30],feature_mat)
    bkmrk = bkmrk + vessel_samples
"""

"""#test featuremat_noscale.py
import featuremat_noscale

scales = np.arange(3,50,5)
img = cv2.imread('../data/color/quek1.bmp')
img = cv2.pyrDown(img)
vessel_bin = np.load('../data/vessels/red/quek1.npy')
vessel_ind = np.nonzero(vessel_bin)
feature_mat_maker = featuremat.FeatureMatMaker(img,scales)
feature_mat_maker_noscale = featuremat_noscale.FeatureMatMaker(img)
non_vessel_ind = feature_mat_maker.getRandInd(vessel_ind)
vessel_feature_mat,non_vessel_feature_mat = feature_mat_maker.getTrainMat(vessel_ind,non_vessel_ind)
vessel_feature_mat2,non_vessel_feature_mat2 = feature_mat_maker_noscale.getTrainMat(vessel_ind,non_vessel_ind)

vessel_feature_mat = vessel_feature_mat.round(8)
vessel_feature_mat2 = vessel_feature_mat2.round(8)
non_vessel_feature_mat = non_vessel_feature_mat.round(8)
non_vessel_feature_mat2 = non_vessel_feature_mat2.round(8)
print np.array_equal(vessel_feature_mat[:,:30],vessel_feature_mat2)
print np.array_equal(non_vessel_feature_mat[:,:30],non_vessel_feature_mat2)
"""

"""# test categorizeInd
#img = cv2.imread('../data/color/quek1.bmp')
img = np.zeros((5,5,3)).astype(np.uint8)
scale = np.arange(3,50,5)
#vessel_bin = np.load('../data/vessels/red/quek1.npy')
vessel_bin = np.random.rand(5,5)<0.2
vessel_ind = np.nonzero(vessel_bin)
feature_mat_maker = featuremat.FeatureMatMaker(img,scale)
non_vessel_ind = feature_mat_maker.getRandInd(vessel_ind)
non_vessel_bin = np.zeros((5,5)).astype(np.bool)
non_vessel_bin[non_vessel_ind] = True
#orient = feature_mat_maker.ridgeOrient()
orient = np.random.rand(5,5)*180-90
(categorized_vessel_ind,
 categorized_non_vessel_ind) = feature_mat_maker.categorizeInd(orient,
                                                               vessel_ind,
                                                               non_vessel_ind)
"""

"""# test FeatureMatMaker.getTrainMat
scale = np.arange(3,50,5)
#img = (np.random.rand(5,5,3)*255).astype(np.uint8)
img = cv2.imread('../data/color/quek1.bmp')
#vessel_bin = np.random.rand(5,5)<0.2
#vessel_ind = np.nonzero(vessel_bin)
vessel_bin = np.load('../data/vessels/red/quek1.npy')
vessel_ind = np.nonzero(vessel_bin)
feature_mat_maker = featuremat.FeatureMatMaker(img,scale)

(vessel_feature_mat,
 non_vessel_feature_mat,
 categorized_vessel_ind,
 categorized_non_vessel_ind) = feature_mat_maker.getTrainMat(vessel_ind)
 
img = featuremat.getScaledImg(img,33)
rotated_vessel_ind_sizes = [166,221,82,49,21,9,13,8,6,8,2,6,7,11,14,25,47,111] # size of categorized vessel ind for each orientation(for quek1.bmp)
(rotated_img,
 rotated_vessel_ind,
 rotated_non_vessel_ind) = feature_mat_maker.rotateImgAndInd(img,
                                                             categorized_vessel_ind[16],
                                                             categorized_non_vessel_ind[16],
                                                             -75)
x = cv2.Sobel(rotated_img[:,:,0],cv2.CV_64F,1,0)
y = cv2.Sobel(rotated_img[:,:,0],cv2.CV_64F,0,1)
xx = cv2.Sobel(rotated_img[:,:,0],cv2.CV_64F,2,0)
yy = cv2.Sobel(rotated_img[:,:,0],cv2.CV_64F,0,2)
xy = cv2.Sobel(rotated_img[:,:,0],cv2.CV_64F,1,1)
xxx = cv2.Sobel(rotated_img[:,:,0],cv2.CV_64F,3,0,ksize=5)
xxy = cv2.Sobel(rotated_img[:,:,0],cv2.CV_64F,2,1,ksize=5)
xyy = cv2.Sobel(rotated_img[:,:,0],cv2.CV_64F,1,2,ksize=5)
yyy = cv2.Sobel(rotated_img[:,:,0],cv2.CV_64F,0,3,ksize=5)

x2 = cv2.Sobel(rotated_img[:,:,1],cv2.CV_64F,1,0)
y2 = cv2.Sobel(rotated_img[:,:,1],cv2.CV_64F,0,1)
xx2 = cv2.Sobel(rotated_img[:,:,1],cv2.CV_64F,2,0)
yy2 = cv2.Sobel(rotated_img[:,:,1],cv2.CV_64F,0,2)
xy2 = cv2.Sobel(rotated_img[:,:,1],cv2.CV_64F,1,1)
xxx2 = cv2.Sobel(rotated_img[:,:,1],cv2.CV_64F,3,0,ksize=5)
xxy2 = cv2.Sobel(rotated_img[:,:,1],cv2.CV_64F,2,1,ksize=5)
xyy2 = cv2.Sobel(rotated_img[:,:,1],cv2.CV_64F,1,2,ksize=5)
yyy2 = cv2.Sobel(rotated_img[:,:,1],cv2.CV_64F,0,3,ksize=5)

x3 = cv2.Sobel(rotated_img[:,:,2],cv2.CV_64F,1,0)
y3 = cv2.Sobel(rotated_img[:,:,2],cv2.CV_64F,0,1)
xx3 = cv2.Sobel(rotated_img[:,:,2],cv2.CV_64F,2,0)
yy3 = cv2.Sobel(rotated_img[:,:,2],cv2.CV_64F,0,2)
xy3 = cv2.Sobel(rotated_img[:,:,2],cv2.CV_64F,1,1)
xxx3 = cv2.Sobel(rotated_img[:,:,2],cv2.CV_64F,3,0,ksize=5)
xxy3 = cv2.Sobel(rotated_img[:,:,2],cv2.CV_64F,2,1,ksize=5)
xyy3 = cv2.Sobel(rotated_img[:,:,2],cv2.CV_64F,1,2,ksize=5)
yyy3 = cv2.Sobel(rotated_img[:,:,2],cv2.CV_64F,0,3,ksize=5)

bkmrk = sum(rotated_vessel_ind_sizes[:16])
bkmrk2 = sum(feature_mat_maker.rotated_non_vessel_ind_sizes[:16])
offset = rotated_vessel_ind_sizes[16]
offset2 = 180
offset3 = feature_mat_maker.rotated_non_vessel_ind_sizes[16]
print np.array_equal(x[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+1])
print np.array_equal(y[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+2])
print np.array_equal(xx[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+3])
print np.array_equal(yy[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+4])
print np.array_equal(xy[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+5])
print np.array_equal(xxx[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+6])
print np.array_equal(xxy[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+7])
print np.array_equal(xyy[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+8])
print np.array_equal(yyy[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+9])

print np.array_equal(x2[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+10+1])
print np.array_equal(y2[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+10+2])
print np.array_equal(xx2[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+10+3])
print np.array_equal(yy2[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+10+4])
print np.array_equal(xy2[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+10+5])
print np.array_equal(xxx2[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+10+6])
print np.array_equal(xxy2[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+10+7])
print np.array_equal(xyy2[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+10+8])
print np.array_equal(yyy2[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+10+9])

print np.array_equal(x3[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+20+1])
print np.array_equal(y3[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+20+2])
print np.array_equal(xx3[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+20+3])
print np.array_equal(yy3[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+20+4])
print np.array_equal(xy3[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+20+5])
print np.array_equal(xxx3[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+20+6])
print np.array_equal(xxy3[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+20+7])
print np.array_equal(xyy3[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+20+8])
print np.array_equal(yyy3[rotated_vessel_ind],vessel_feature_mat[bkmrk:bkmrk+offset,offset2+20+9])

print np.array_equal(x[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+1])
print np.array_equal(y[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+2])
print np.array_equal(xx[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+3])
print np.array_equal(yy[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+4])
print np.array_equal(xy[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+5])
print np.array_equal(xxx[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+6])
print np.array_equal(xxy[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+7])
print np.array_equal(xyy[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+8])
print np.array_equal(yyy[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+9])

print np.array_equal(x2[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+10+1])
print np.array_equal(y2[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+10+2])
print np.array_equal(xx2[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+10+3])
print np.array_equal(yy2[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+10+4])
print np.array_equal(xy2[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+10+5])
print np.array_equal(xxx2[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+10+6])
print np.array_equal(xxy2[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+10+7])
print np.array_equal(xyy2[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+10+8])
print np.array_equal(yyy2[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+10+9])

print np.array_equal(x3[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+20+1])
print np.array_equal(y3[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+20+2])
print np.array_equal(xx3[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+20+3])
print np.array_equal(yy3[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+20+4])
print np.array_equal(xy3[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+20+5])
print np.array_equal(xxx3[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+20+6])
print np.array_equal(xxy3[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+20+7])
print np.array_equal(xyy3[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+20+8])
print np.array_equal(yyy3[rotated_non_vessel_ind],non_vessel_feature_mat[bkmrk2:bkmrk2+offset3,offset2+20+9])
"""

"""# test efficient version of FeatureMatMaker.getTrainMat
scale = np.arange(3,50,5)
img = cv2.imread('../data/color/quek1.bmp')
vessel_bin = np.load('../data/vessels/red/quek1.npy')
vessel_ind = np.nonzero(vessel_bin)
feature_mat_maker = featuremat.FeatureMatMaker(img,scale)

start = time.time()
(vessel_feature_mat,
 non_vessel_feature_mat) = feature_mat_maker.getTrainMat(vessel_ind)
print time.time() - start

start = time.time()
(vessel_feature_mat2,
 non_vessel_feature_mat2) = feature_mat_maker.getTrainMat2(vessel_ind,feature_mat_maker.non_vessel_ind)
print time.time() - start

np.array_equal(vessel_feature_mat,vessel_feature_mat2)
np.array_equal(non_vessel_feature_mat,non_vessel_feature_mat2)
"""

"""# test FeatureMatMaker.rotateImg
scale = np.arange(3,50,5)
img = cv2.imread('../data/color/quek1.bmp')
img = cv2.pyrDown(img)
feature_mat_maker = featuremat.FeatureMatMaker(img,scale)
orient = feature_mat_maker.ridgeOrient()
categorized = feature_mat_maker.categorizePixels(orient)
rotated_img,rotated_is_angle = feature_mat_maker.rotateImg(img,categorized[1],75)
cv2.imwrite('../data/junk/test1.jpg',rotated_img)
cv2.imwrite('../data/junk/test2.jpg',rotated_is_angle)
"""

"""# orientation histogram
import matplotlib.pyplot as plt
import featuremat_normalized_features
scale = np.arange(3,50,5)
img = cv2.imread('../data/orientation/direction4.bmp')
img = cv2.pyrDown(img)
feature_mat_maker = featuremat_normalized_features.FeatureMatMaker(img,scale)
rotated_img,dummy = feature_mat_maker.rotateImg(img,img[:,:,2],-50)
orient = featuremat_normalized_features.ridgeOrient(img)
orient = np.around(orient[np.logical_not(np.isnan(orient))],-1)
bin = (orient == -90).astype(np.uint8)
orient = orient + bin*180
n, bins, patches = plt.hist(orient, 18, normed=1, facecolor='green', alpha=0.75)
"""

"""# normalize image orientation
import featuremat_normalized_features
scale = np.arange(3,50,5)
img = cv2.imread('../data/orientation/direction2.bmp')
img = cv2.pyrDown(img)
feature_mat_maker = featuremat_normalized_features.FeatureMatMaker(img,scale)
orient_img = featuremat_normalized_features.ridgeOrient(img)
orient_img = np.around(orient_img[np.logical_not(np.isnan(orient_img))],-1)
bin = (orient_img == -90).astype(np.uint8)
orient_img = orient_img + bin*180

u,ind = np.unique(orient_img,return_inverse=True)
orient = u[np.argmax(np.bincount(ind))]

rotated_img,dummy = featuremat_normalized_features.rotateImg(img,img[:,:,2],orient)
cv2.imshow('stuff',rotated_img)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('../data/junk/direction2.jpg',rotated_img)
"""

"""# test numpy.hstack
random=[]
for i in range(10):
    random.append(np.around(np.random.rand(40000)*100))
foo = np.array([]).reshape(0,40000)

start = time.time()
for stuff in random:
    foo = np.vstack((foo,stuff))
print time.time()-start

start = time.time()
foo2 = np.vstack((random[0],random[1],random[2],random[3],random[4],random[5],random[6],random[7],random[8],random[9]))
print time.time()-start
print np.array_equal(foo,foo2)
"""