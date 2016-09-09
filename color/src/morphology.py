# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 07:14:54 2016

@author: queky
"""
import numpy as np
import itertools
import cv2

def hitOrMiss(img,struct_ele):
    assert np.size(np.setdiff1d(img,np.array([0,255]))) == 0
    img = img/255
    assert np.size(np.setdiff1d(struct_ele,np.array([-1,0,1]))) == 0 # ensure array of -1,0,1
    assert np.shape(struct_ele) == (3,3)
    struct_fore = (struct_ele == 1).astype(np.uint8)
    struct_fore_sum = np.sum(struct_fore)
    struct_back = (struct_ele == 0).astype(np.uint8)
    fore_hits = cv2.filter2D(img,-1,struct_fore,borderType=cv2.BORDER_CONSTANT) == struct_fore_sum
    back_hits = cv2.filter2D(img,-1,struct_back,borderType=cv2.BORDER_CONSTANT) == 0
    return fore_hits*back_hits