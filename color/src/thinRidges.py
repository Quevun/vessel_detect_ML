# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 07:27:11 2016

@author: queky
"""
import cv2
import numpy as np
import skimage.morphology
import morphology
import copy

def getCoords(event,x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(params[1]) < 2 and len(params[2]) < 2:
            params[1].append(x)
            params[2].append(y)
        if len(params[1]) == 2 and len(params[2]) == 2:
            minX = min(params[1])
            maxX = max(params[1])
            minY = min(params[2])
            maxY = max(params[2])
            sizeX = maxX - minX + 1
            sizeY = maxY - minY + 1
            remove_patch = copy.copy(params[0][minY:maxY+1,minX:maxX+1])
            params[3].append((minX,maxX,minY,maxY,remove_patch))
            params[0][minY:maxY+1,minX:maxX+1] = np.zeros((sizeY,sizeX))
            del params[1][:]
            del params[2][:]

def manualRemove(img):
    params = (img,list(),list(),list())
    key = None
    cv2.imshow('Manual White Pixel Removal',img)
    cv2.setMouseCallback('Manual White Pixel Removal', getCoords,params)
    while not (key == 13): # End loop when 'Enter' is pressed
        key = cv2.waitKey()
        if key == 122:  #When 'z' is pressed, undo previous removal
            print 'undoing...'
            minX,maxX,minY,maxY,removed = params[3].pop()
            img[minY:maxY+1,minX:maxX+1] = removed
            cv2.imshow('Manual White Pixel Removal',img)
        elif key == 117:    # When 'u' is pressed update image
            print 'updating image...'            
            cv2.imshow('Manual White Pixel Removal',img)
    cv2.destroyAllWindows()

filename = 'yokoyama4'
img = np.load('../data/eigen/red/'+filename+'.npy')
skel = skimage.morphology.skeletonize(img>0)
branch_len = 20
pruned = skel.astype(np.uint8)*255
struc_ele1 = np.array([[-1,0,0],[1,1,0],[-1,0,0]])
struc_ele2 = np.array([[1,0,0],[0,1,0],[0,0,0]])
struc_ele_tuple = (struc_ele1,np.rot90(struc_ele1,1),np.rot90(struc_ele1,2),np.rot90(struc_ele1,3),
                   struc_ele2,np.rot90(struc_ele2,1),np.rot90(struc_ele2,2),np.rot90(struc_ele2,3))

for i in range(branch_len):
    end_points = np.zeros(np.shape(skel)).astype(np.bool)    
    for struc_ele in struc_ele_tuple:
        end_points = end_points + morphology.hitOrMiss(pruned,struc_ele)
    pruned = pruned * np.invert(end_points)
 
struc_ele3 = np.array([[0,0,0],[0,1,0],[0,0,0]])
single_points = morphology.hitOrMiss(pruned,struc_ele3)
pruned = pruned * np.invert(single_points)
"""
color = cv2.imread('../data/color/'+filename+'.bmp')
color_small = cv2.pyrDown(color)
cv2.imwrite('../data/color/small/red/'+filename+'.jpg',color_small)
color_small[:,:,0] = color_small[:,:,0]*(pruned==0)
color_small[:,:,1] = color_small[:,:,1]*(pruned==0)
color_small[:,:,2] = color_small[:,:,2]*(pruned==0)
cv2.imwrite('../data/color/small/red/'+filename+'_vessels.jpg',color_small)
"""
manualRemove(pruned)
np.save('../data/vessels/major_vessels_only/'+filename,pruned)