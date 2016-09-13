# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 14:27:44 2016

@author: Quek Yu Yang
"""
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np


def plotImg(img):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = range(np.size(img,1))
    y = range(np.size(img,0))
    X, Y = np.meshgrid(x, y)
    plt.gca().invert_yaxis()
    ax.plot_surface(X,Y,img)
"""
img = cv2.imread('input/test.bmp',cv2.IMREAD_GRAYSCALE)
img = cv2.pyrDown(img)
img = hlpr.getScaledImg(img,225)
plotImg(img)
"""


def getCoord(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print (x,y)
 
"""
img = cv2.imread('input/marker.bmp',cv2.IMREAD_GRAYSCALE)
cv2.namedWindow("image")
cv2.setMouseCallback("image", getCoord)
cv2.imshow('image',img)
cv2.waitKey()
cv2.destroyAllWindows()
"""

def getNeighbourCoords(coord,size): # coord: (x,y)
    assert size%2 == 1
    coords = np.zeros((size,size,2))
    up_left = (coord[0] - (size-1)/2, coord[1] - (size-1)/2)
    yOffset = np.repeat(np.array(range(size))[np.newaxis,:].T,size,1)
    xOffset = np.repeat(np.array(range(size))[np.newaxis,:],size,0)
    coords[:,:,1] = np.ones((size,size))*up_left[1] + yOffset
    coords[:,:,0] = np.ones((size,size))*up_left[0] + xOffset
    return coords.reshape(size**2,2).astype(np.int)

def plotRidgeStrAlongScale(cuboid,coords):
    for coord in coords:
        y = cuboid[coord[1],coord[0],:]
        plt.plot(y)
        
def plotAlongAxis(axis,mat,index):
    if axis == 0:
        y = mat[200:300,index]
        plt.plot(y)
    elif axis == 1:
        y = mat[index,:]
        plt.plot(y)
        
"""
img = cv2.imread('input/IR3/test7.bmp',cv2.IMREAD_GRAYSCALE)
for scale in range(5):
    scaled = hlpr.ScaledImage(img,scale)
    anaFunc.plotAlongAxis(0,scaled.getImg,452)
"""