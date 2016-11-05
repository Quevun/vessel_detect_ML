# -*- coding: utf-8 -*-
"""
Created on Thu Sep 08 09:46:31 2016
@author: Quek Yu Yang
"""
import numpy as np
import cv2
import sys

class FeatureMatMaker(object):
    def __init__(self,img,scales):
        assert np.size(img,2) == 3
        assert img.dtype == 'uint8'
        self.img = img
        self.scales = scales
        self.num_features = 30
        self.angle = range(-85,90,10)
        
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
            feature_cube[:,:,nf*i+1:nf*i+nf+1] = np.concatenate((first,x1,y1,xx1,yy1,xy1,xxx1,xxy1,xyy1,yyy1,
                                                                 second,x2,y2,xx2,yy2,xy2,xxx2,xxy2,xyy2,yyy2,
                                                                 third,x3,y3,xx3,yy3,xy3,xxx3,xxy3,xyy3,yyy3,),
                                                                 axis=2)
            
        feature_mat = np.reshape(feature_cube,(shape[0]*shape[1],shape[2]),'F')
        feature_mat = self.featureScale(feature_mat)
        feature_mat[:,0] = 1  # First element of each feature vector is 1
        return feature_mat
            
    def getTrainMat(self,vessel_ind,non_vessel_ind = None):
        #######################
        #   Initialization
        self.vessel_ind = vessel_ind
        nf = self.num_features
        num_scales = len(self.scales)
        vessel_feature_mat = np.array([]).reshape((0,nf))
        
        if non_vessel_ind == None:
            non_vessel_ind = self.getRandInd()
        non_vessel_feature_mat = np.array([]).reshape((0,nf))
        #######################        
        
        orient = self.ridgeOrient()
        (categorized_vessel_ind,
         categorized_non_vessel_ind) = self.categorizeInd(orient,
                                                          vessel_ind,
                                                          non_vessel_ind)
        ####################################
        # Make feature matrix
        vessel_feature_submat_list = []
        self.rotated_vessel_ind_list = []
        for i in range(num_scales):
            scaled = getScaledImg(self.img,self.scales[i])
            vessel_feature_submat = np.array([]).reshape((0,nf))
            non_vessel_feature_submat = np.array([]).reshape((0,nf))
            for j in range(len(self.angle)):
                
                (rotated_img,
                 rotated_vessel_ind,
                 rotated_non_vessel_ind) = self.rotateImgAndInd(scaled,
                                                                categorized_vessel_ind[j],# do something about empty ind
                                                                categorized_non_vessel_ind[j],
                                                                self.angle[len(self.angle)-j-1])
                if i == 0:#################
                    self.rotated_vessel_ind_list.append(rotated_vessel_ind)################
                (vessel_deriv_mat,
                 non_vessel_deriv_mat) = self.getDerivMat(rotated_img,
                                                          rotated_vessel_ind,
                                                          rotated_non_vessel_ind)

                vessel_feature_submat = np.concatenate((vessel_feature_submat,
                                                     vessel_deriv_mat.T),0)
                non_vessel_feature_submat = np.concatenate((non_vessel_feature_submat,
                                                         non_vessel_deriv_mat.T),0)
            if not 'vessel_feature_mat_ax0_size' in locals():
                vessel_feature_mat_ax0_size = vessel_feature_submat.shape[0]
            if not 'non_vessel_feature_mat_ax0_size' in locals():
                non_vessel_feature_mat_ax0_size = non_vessel_feature_submat.shape[0]
            vessel_feature_submat_list.append(vessel_feature_submat)
        vessel_feature_mat = np.array([]).reshape((vessel_feature_mat_ax0_size,0))
        for i in vessel_feature_submat_list:
            vessel_feature_mat = np.concatenate((vessel_feature_mat,i),1)

        ####################################
        """
        scaled_features = self.featureScale(np.concatenate((vessel_feature_mat,
                                                           non_vessel_feature_mat),axis=0))
        vessel_feature_mat = scaled_features[:vessel_feature_mat_ax0_size,:]
        non_vessel_feature_mat = scaled_features[-non_vessel_feature_mat_ax0_size:,:]
        """    
        return vessel_feature_mat,non_vessel_feature_mat,categorized_vessel_ind,categorized_non_vessel_ind
            
    def getRandInd(self,vessel_ind=None):
        if vessel_ind == None:
            vessel_ind = self.vessel_ind
        y = vessel_ind[0][np.newaxis,:]
        x = vessel_ind[1][np.newaxis,:]
        vessel_ind_struc = np.concatenate((y,x),axis=0).flatten('F')
        vessel_ind_struc = vessel_ind_struc.view([('y',np.int64),
                                                  ('x',np.int64)]) # structured array with (y,x) elements
        sample_size = np.around(vessel_ind[0].size*3)
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
        
        x3 = cv2.Sobel(third,cv2.CV_64F,1,0)
        y3 = cv2.Sobel(third,cv2.CV_64F,0,1)
        xx3 = cv2.Sobel(third,cv2.CV_64F,2,0)
        yy3 = cv2.Sobel(third,cv2.CV_64F,0,2)
        xy3 = cv2.Sobel(third,cv2.CV_64F,1,1)
        xxx3 = cv2.Sobel(third,cv2.CV_64F,3,0,ksize=5)
        xxy3 = cv2.Sobel(third,cv2.CV_64F,2,1,ksize=5)
        xyy3 = cv2.Sobel(third,cv2.CV_64F,1,2,ksize=5)
        yyy3 = cv2.Sobel(third,cv2.CV_64F,0,3,ksize=5)
        
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
        
        u3 = third[vessel_ind][np.newaxis,:]
        ux3 = x3[vessel_ind][np.newaxis,:]
        uy3 = y3[vessel_ind][np.newaxis,:]
        uxx3 = xx3[vessel_ind][np.newaxis,:]
        uyy3 = yy3[vessel_ind][np.newaxis,:]
        uxy3 = xy3[vessel_ind][np.newaxis,:]
        uxxx3 = xxx3[vessel_ind][np.newaxis,:]
        uxxy3 = xxy3[vessel_ind][np.newaxis,:]
        uxyy3 = xyy3[vessel_ind][np.newaxis,:]
        uyyy3 = yyy3[vessel_ind][np.newaxis,:]
                                     
        vessel_deriv_mat = np.concatenate((u1,ux1,uy1,uxx1,uyy1,uxy1,uxxx1,uxxy1,uxyy1,uyyy1,
                                           u2,ux2,uy2,uxx2,uyy2,uxy2,uxxx2,uxxy2,uxyy2,uyyy2,
                                           u3,ux3,uy3,uxx3,uyy3,uxy3,uxxx3,uxxy3,uxyy3,uyyy3),axis=0)
                                           
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
        
        u3 = third[non_vessel_ind][np.newaxis,:]
        ux3 = x3[non_vessel_ind][np.newaxis,:]
        uy3 = y3[non_vessel_ind][np.newaxis,:]
        uxx3 = xx3[non_vessel_ind][np.newaxis,:]
        uyy3 = yy3[non_vessel_ind][np.newaxis,:]
        uxy3 = xy3[non_vessel_ind][np.newaxis,:]
        uxxx3 = xxx3[non_vessel_ind][np.newaxis,:]
        uxxy3 = xxy3[non_vessel_ind][np.newaxis,:]
        uxyy3 = xyy3[non_vessel_ind][np.newaxis,:]
        uyyy3 = yyy3[non_vessel_ind][np.newaxis,:]

        non_vessel_deriv_mat = np.concatenate((u1,ux1,uy1,uxx1,uyy1,uxy1,uxxx1,uxxy1,uxyy1,uyyy1,
                                               u2,ux2,uy2,uxx2,uyy2,uxy2,uxxx2,uxxy2,uxyy2,uyyy2,
                                               u3,ux3,uy3,uxx3,uyy3,uxy3,uxxx3,uxxy3,uxyy3,uyyy3),axis=0)
        return vessel_deriv_mat,non_vessel_deriv_mat
        
    def rotateImgAndInd(self,img,vessel_ind,non_vessel_ind,rot_angle):
        diag_len = round(np.sqrt(img.shape[0]**2 + img.shape[1]**2))
        x_pad = round((diag_len - img.shape[1] + 5)/2)*2
        y_pad = round((diag_len - img.shape[0] + 5)/2)*2
        padded_img = np.zeros((img.shape[0]+y_pad,img.shape[1]+x_pad,img.shape[2]))
        padded_vessel_bin = np.zeros((img.shape[0]+y_pad,img.shape[1]+x_pad))
        padded_non_vessel_bin = np.copy(padded_vessel_bin)
        x_mid = round(padded_img.shape[1]/2)
        y_mid = round(padded_img.shape[0]/2)
        
        padded_img[y_pad/2:-y_pad/2,x_pad/2:-x_pad/2,:] = img
        padded_img = padded_img.astype(np.uint8)
        
        vessel_bin = np.zeros((self.img.shape[0],self.img.shape[1])).astype(np.uint8)
        vessel_bin[vessel_ind] = 255
        padded_vessel_bin[y_pad/2:-y_pad/2,x_pad/2:-x_pad/2] = vessel_bin
        
        non_vessel_bin = np.zeros((self.img.shape[0],self.img.shape[1])).astype(np.uint8)
        non_vessel_bin[non_vessel_ind] = 255
        padded_non_vessel_bin[y_pad/2:-y_pad/2,x_pad/2:-x_pad/2] = non_vessel_bin
        
        try:
            rot_angle.shape
            multiple_angles = True
        except AttributeError:
            multiple_angles = False
        
        if multiple_angles:
            rotated_img = [img]
            rotated_vessel_ind = [vessel_ind]
            rotated_non_vessel_ind = [non_vessel_ind]
            for angle in rot_angle:
                rot_mat = cv2.getRotationMatrix2D((x_mid,y_mid),angle,1)
                
                # rotate img
                temp = np.zeros((padded_img.shape))
                temp[:,:,0] = cv2.warpAffine(padded_img[:,:,0],rot_mat
                                            ,(padded_img.shape[1],padded_img.shape[0]))
                temp[:,:,1] = cv2.warpAffine(padded_img[:,:,1],rot_mat
                                            ,(padded_img.shape[1],padded_img.shape[0]))
                temp[:,:,2] = cv2.warpAffine(padded_img[:,:,2],rot_mat
                                            ,(padded_img.shape[1],padded_img.shape[0]))
                temp = temp.astype(np.uint8)
                rotated_img.append(temp)
                
                # rotate vessel coordinates
                temp = cv2.warpAffine(padded_vessel_bin,rot_mat
                                     ,(padded_img.shape[1],padded_img.shape[0])
                                     ,flags=cv2.INTER_NEAREST).astype(np.uint8)
                temp = np.nonzero(temp)
                rotated_vessel_ind.append(temp)
                
                # rotate non vessel coordinates
                temp = cv2.warpAffine(padded_non_vessel_bin,rot_mat
                                     ,(padded_img.shape[1],padded_img.shape[0])
                                     ,flags=cv2.INTER_NEAREST).astype(np.uint8)
                temp = np.nonzero(temp)
                rotated_non_vessel_ind.append(temp)
                
        else:
            rot_mat = cv2.getRotationMatrix2D((x_mid,y_mid),rot_angle,1)
                
            # rotate img
            rotated_img = np.zeros((padded_img.shape))
            rotated_img[:,:,0] = cv2.warpAffine(padded_img[:,:,0],rot_mat
                                        ,(padded_img.shape[1],padded_img.shape[0]))
            rotated_img[:,:,1] = cv2.warpAffine(padded_img[:,:,1],rot_mat
                                        ,(padded_img.shape[1],padded_img.shape[0]))
            rotated_img[:,:,2] = cv2.warpAffine(padded_img[:,:,2],rot_mat
                                        ,(padded_img.shape[1],padded_img.shape[0]))
            rotated_img = rotated_img.astype(np.uint8)
            
            # rotate vessel coordinates
            rotated_vessel_ind = cv2.warpAffine(padded_vessel_bin,rot_mat,
                                                (padded_img.shape[1],padded_img.shape[0]),
                                                flags=cv2.INTER_NEAREST).astype(np.uint8)
            rotated_vessel_ind = np.nonzero(rotated_vessel_ind)
            
            # rotate non vessel coordinates
            rotated_non_vessel_ind = cv2.warpAffine(padded_non_vessel_bin,rot_mat,
                                                    (padded_img.shape[1],padded_img.shape[0]),
                                                    flags=cv2.INTER_NEAREST).astype(np.uint8)
            rotated_non_vessel_ind = np.nonzero(rotated_non_vessel_ind)
        
        return rotated_img,rotated_vessel_ind,rotated_non_vessel_ind
        
    def ridgeOrient(self):
        img = self.img[:,:,2]
        img = getScaledImg(img,10)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
        orient = np.arctan(sobely/sobelx)
        orient = np.nan_to_num(orient)
        orient = np.rad2deg(orient)
        return orient
        
    # Categorize indices according to orientation
    def categorizeInd(self,orient,vessel_ind,non_vessel_ind):
        angle = self.angle
        vessel_orient = orient[vessel_ind]
        non_vessel_orient = orient[non_vessel_ind]
        categorized_vessel_ind = []
        categorized_non_vessel_ind = []
        #vessel_orient = np.random.rand(10)*180-90
        for i in range(len(angle)):
            is_angle = ((vessel_orient >= angle[i]-5)*
                        (vessel_orient < angle[i]+5))
            axis0_ind = vessel_ind[0][np.nonzero(is_angle)]
            axis1_ind = vessel_ind[1][np.nonzero(is_angle)]
            categorized_vessel_ind.append((axis0_ind,axis1_ind))
            
            is_angle = ((non_vessel_orient >= angle[i]-5)*
                        (non_vessel_orient < angle[i]+5))
            axis0_ind = non_vessel_ind[0][np.nonzero(is_angle)]
            axis1_ind = non_vessel_ind[1][np.nonzero(is_angle)]
            categorized_non_vessel_ind.append((axis0_ind,axis1_ind))
            
        return categorized_vessel_ind,categorized_non_vessel_ind
        
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