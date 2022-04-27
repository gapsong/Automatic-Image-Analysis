#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Matern
"""

from unittest import result
import numpy as np
import cv2
import utils

def convert_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def binarizeImage(img, thresh):
    """
    Given a coloured image and a threshold binarizes the image.
    Values below thresh are set to 0. All other values are set to 255
    """
    grayscale = convert_to_grayscale(img)
    result = np.array(grayscale)

    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            if result[i][j] >= thresh:
                result[i][j] = 255
            else:
                result[i][j] = 0
    return result


def smoothImage(img):    
    """
    Given a coloured image apply a blur on the image, e.g. Gaussian blur
    """
    result = cv2.GaussianBlur(img,(5,51),cv2.BORDER_DEFAULT)
    return result

def doSomething(img):
    """
    Given a coloured image apply any image manipulation. Be creative!
    """
    result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return result


def processImage(img):
    """
    Given an coloured image applies the implemented smoothing and binarization.
    """
    img = smoothImage(img)
    img = binarizeImage(img, 125)
    return img


if __name__=="__main__":
    img = cv2.imread("test.jpg")
    utils.show(img)
    
    img1 = smoothImage(img)
    utils.show(img1)
    cv2.imwrite("result1.jpg", img1)
    
    img2 = binarizeImage(img, 125)
    utils.show(img2)
    cv2.imwrite("result2.jpg", img2)
   
    img3 = processImage(img)
    utils.show(img3)
    cv2.imwrite("result3.jpg", img3)
    
    img4 = doSomething(img)
    utils.show(img4)
    cv2.imwrite("result4.jpg", img4)
