# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 22:18:51 2021

@author: 41605
"""
import cv2
import numpy as np
import math

def histogram_stretching(img):
    img_stretch = np.copy(img)
    maxVal = np.max(img_stretch) 
    minVal = np.min(img_stretch)
    dynamic = maxVal-minVal
    h, w, ch = img_stretch.shape
    for row in range(h):
        for col in range(w):
            b, g, r = img_stretch[row, col]
            b = ((b-minVal)/dynamic)*255
            g = ((g-minVal)/dynamic)*255
            r = ((r-minVal)/dynamic)*255
            img_stretch[row, col] = [b, g, r]
    return img_stretch

def power_law_transformation(img, c, gamma):
    img_powerlaw = np.copy(img)
    h, w, ch = img_powerlaw.shape
    for row in range(h):
        for col in range(w):
            b, g, r = img_powerlaw[row, col]
            b = c*math.pow(b, gamma)
            g = c*math.pow(g, gamma)
            r = c*math.pow(r, gamma)
            img_powerlaw[row, col] = [b, g, r]
    return img_powerlaw 

def median_filter(img, filter_size):
    h,w,ch = img.shape
    pad = filter_size//2
    img_medianfilter = np.zeros((h + 2*pad,w + 2*pad,ch),dtype=np.float)
    img_medianfilter[pad:pad+h,pad:pad+w] = img.copy().astype(np.float)
    tmp = img_medianfilter.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(ch):
                img_medianfilter[pad+y,pad+x,ci] = np.median(tmp[y:y+filter_size,x:x+filter_size,ci])  
    img_medianfilter = img_medianfilter[pad:pad+h,pad:pad+w].astype(np.uint8)
    return img_medianfilter

image_all = [r'peppers.pgm',r'Building.pgm',r'MRI.pgm']

for each_image in image_all:
    # load image
    img=cv2.imread(each_image)
    # display input image
    cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("input", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # display histogram stretching image
    img_stretch = histogram_stretching(img)
    cv2.namedWindow("output1", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("output1", img_stretch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # display power law transformation image
    img_powerlaw = power_law_transformation(img_stretch,1,1.05)
    cv2.namedWindow("output2", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("output2", img_powerlaw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # display median filter image
    img_medianfilter = median_filter(img_powerlaw,3)
    cv2.namedWindow("output3", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("output3", img_medianfilter)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
