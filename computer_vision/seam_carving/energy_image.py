## File name: energy_image.py
## Project specifics: ECE 5554 Computer Vision - PS1
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Tuesday, September 8, 2015
## 
## Code synopsis: this file contains a python function to compute the energy at each pixel using the magnitude of the x & y gradients (ref. Avidan & Shamir: Seam Carving for Content-Aware Image Resizing).
##
## Inputs - im: MxNx3 matrix of datatype uint8 (output of imread function)
## Outputs - energyImage: 2D matrix of datatype double

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage as nd

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.2989,0.5870,0.1140])

def energy_image(im_in,type):
    im = np.copy(im_in)
    im = rgb2gray(im)
    im.astype(np.float64)
    [im_x,im_y] = np.gradient(im)
    im_out = np.hypot(im_x,im_y)
    if type == 'HORIZONTAL':
        im_out[:,0] = im_out[:,1]
        im_out[:,-1] = im_out[:,-2]
        im_out[0,:] = im_out[1,:]
        im_out[-1,:] = im_out[-2,:]

    elif type == 'VERTICAL':
        im_out[0,:] = im_out[1,:]
        im_out[-1,:] = im_out[-2,:]
        im_out[:,0] = im_out[:,1]
        im_out[:,-1] = im_out[:,-2]
    else:
        print('no type given')
    
    return im_out
