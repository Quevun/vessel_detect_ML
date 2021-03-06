# -*- coding: utf-8 -*-
"""
Created on Fri Jan 06 10:51:36 2017

@author: Quek Yu Yang
"""

import cv2
import numpy as np
import findBloodVessels
import time

filename = 'video2.avi'
cap = cv2.VideoCapture('../data/video/'+filename)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('../data/video/output2.avi',fourcc, 20.0, (360,240))

skip = 100
for i in range(skip):
    cap.grab()

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    frame = cv2.pyrDown(frame)
    vessel_bin = findBloodVessels.findBloodVessels(frame)
    
    vessel_bin = np.invert(vessel_bin.astype(np.bool))
    vessel_bin = vessel_bin[:,:,np.newaxis].repeat(3,axis=2)
    marked = frame*vessel_bin
    cv2.imshow('stuff',marked)
    out.write(marked)
    
    if cv2.waitKey(25) == ord('q'):
        break
    print time.time() - start
    
cap.release()
out.release()
cv2.destroyAllWindows()