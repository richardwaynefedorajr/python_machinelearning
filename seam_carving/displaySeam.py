## File name: displaySeam.py
## Project specifics: ECE 5554 Computer Vision - PS1
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Tuesday, September 8, 2015
## 
## Code synopsis: this file contains a python function to display a seam on top of an image
##
## Inputs  - im: image of type jpg
##         - seam: vector containing indices of seam
##         - type: string 'HORIZONTAL' or 'VERTICAL' to define the type of seam
## Outputs - display of input image with seam plotted on top of it

import numpy as np

def displaySeam(im_in,seam,type):
    
    im = np.copy(im_in)
    if type == 'HORIZONTAL':
        for i in range(len(seam)):
            im[seam[i],i,0] = 255
            im[seam[i],i,1] = im[seam[i],i,2] = 0
    elif type == 'VERTICAL':
        for i in range(len(seam)):
            im[i,seam[i],0] = 255
            im[i,seam[i],1] = im[i,seam[i],2] = 0
    else:
        print("No type given")

    return im
