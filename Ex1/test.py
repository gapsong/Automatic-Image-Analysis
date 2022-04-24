#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:55:26 2021

@author: semjon
"""

import ex1
import cv2
import utils
from skimage.metrics import structural_similarity as ssim
import numpy as np
from matplotlib import pyplot as plt


def pixelVariation(img):
    
    if len(img.shape) == 2:
        H,W = img.shape
        img = img.reshape(H,W,1)
    
    H,W,C = img.shape

    Z = np.zeros((H,W,C))
    for c in range(C):
        for row in range(1,H-1):
            for col in range(1,W-1):
                Z[row,col,c] = np.abs((img[row-1:row+1,col-1:col+1,c] - img[row,col,c])).max()
    return Z

def smoothMetric(img,ref):
    A = pixelVariation(img)
    B = pixelVariation(ref)

    Z1 = A.mean()
    Z2 = B.mean()
    
    return abs(Z1-Z2)/Z2


def checkSmoothing(img):
    
    d = smoothMetric(ex1.smoothImage(img),img)
    if d > 0.2:
        print("Smoothing seems fine")
        return True
    else:
        print("Not enough smoothing. Try to increase the blurring effect of your method")
        return False


def checkBinarization(img):
    H,W,C = img.shape
    black = np.zeros((H,W))
    white = np.ones((H,W))*255
    
    low, high = img.min(), img.max()
    
    binary = ex1.binarizeImage(img,125)
    
    if binary.shape != (H,W):
        print("There seems to be a problem with the size of your binary image.")
        return False

    if high != 255:
        print("The binary images should only contain values with either 0 or 255.")
        return False
    
    if low != 0:
        print("The binary images should only contain values with either 0 or 255")     
        return False

    if (np.equal(black, binary)).all():
        print("Your binary image is completely black")
        return False        
    
    if (np.equal(white, binary)).all():
        print("Your binary image is completely white")    
        return False        
    
    print("Binarization seems fine")
    return True

def checkDoSomething(img):
    ref = img.copy()
    
    result = ex1.doSomething(img)
    black = np.zeros(result.shape)
    white = np.ones(result.shape)*255
    
    c = result[0,0,0]
    
    
    if ref.shape != result.shape:
        print("The image shape does not match the original image shape")
        return False             
    
    if np.equal(result, c).all():
        print("Your image has a constant value everywhere. Please be more creative!")
        return False               
    
    if np.equal(result, ex1.smoothImage(img)).all():
        print("You only smoothed the image. Please be more creative!")
        return False           
    d = ssim(ref, result, multichannel=True)     
    if d > 0.9:
        print("The result looks too similar: {:.2f}% SSIM score. Manipulate the image even further!".format(d*100))
        return False
    
    print("DoSomething seems fine")
    return True
    
if __name__=="__main__":
    img = cv2.imread("test.jpg")

    points = 0
    points += checkSmoothing(img)
    points += checkBinarization(img)
    points += checkDoSomething(img)
    print("Result:{}/3".format(points))
    
