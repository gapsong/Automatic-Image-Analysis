#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Matern
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np

def show(img):
    """
    This method takes an image in numpy format and displays it.

    Parameters
    ----------
    img : a numpy array describing an image

    Returns
    -------
    None.

    """
    
    shape = img.shape
    if len(shape) == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
    if len(shape) == 2:
        plt.imshow(img, cmap='gray')
        plt.show()


def rotateAndScale(img, angle, scale):
    """
    Rotate and scale an image

    Parameters
    ----------
    img : ndarray
        an image
    angle : float
        angle given in degrees
    scale : float
        scaling of the image

    Returns
    -------
    result : ndarray
        a distorted image

    """
    
    h, w = img.shape
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, scale)

    corners = np.array([[0, 0, 1],[0, h, 1], [w, 0, 1], [w, h, 1]]).T
    corners = M @ corners
    
    shift = corners.min(1)
    M[:,2]-= shift    
    
    b = corners.max(1)-corners.min(1)
    result = cv2.warpAffine(img, M, (int(b[0]),int(b[1])))
    return result

def calcDirectionalGrad(img):
    """
    Computes the gradients in x- and y-direction.
    The resulting gradients are stored as complex numbers.

    Parameters
    ----------
    img : ndarray
        an image

    Returns
    -------
    ndarray
        The array is stored in the following format: grad_x+ i*grad_y
    """
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    
    return sobelx + 1.0j*sobely


def circularShift(img, dx, dy):
    """
    Performs a circular shift and puts the new origin into position (dx,dy)

    Parameters
    ----------
    img : ndarray
        an image
    dx : int
        x coordinate
    dy : int
        y coordinate

    Returns
    -------
    result : ndarray
        image with new center

    """
    img = img.copy()
    result = np.zeros_like(img)
    H,W = img.shape
    
    result[:-dy,:-dx] = img[dy:,dx:]
    result[:-dy,-dx:] = img[dy:,:dx]
    result[-dy:,:-dx] = img[:dy,dx:]
    result[-dy:,-dx:] = img[:dy,:dx]

    return result
