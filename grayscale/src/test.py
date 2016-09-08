# -*- coding: utf-8 -*-
"""
Created on Fri Sep 02 08:15:33 2016

@author: queky
"""
import numpy as np

# test makeTrainSet.py
Ix = (np.random.rand(3,3)*100).astype(np.uint8)
Iy = (np.random.rand(3,3)*100).astype(np.uint8)
Ixx = (np.random.rand(3,3)*100).astype(np.uint8)
Iyy = (np.random.rand(3,3)*100).astype(np.uint8)
Ixy = (np.random.rand(3,3)*100).astype(np.uint8)

index = (np.array([0,1,2]),np.array([0,2,1]))