# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 15:58:10 2016

@author: Quek Yu Yang
"""

import scipy.io
import numpy as np
import cv2

mat_content = scipy.io.loadmat('../data/NN_param.mat')
theta1 = mat_content['Theta1']
theta2 = mat_content['Theta2']
