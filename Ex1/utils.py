#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Simon Matern
"""
import cv2
from matplotlib import pyplot as plt

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
