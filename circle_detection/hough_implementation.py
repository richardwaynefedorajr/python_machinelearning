## File name: hough_implementation.py
## Project specifics: (package, workspace, etc.) - ECE 5554 Computer Vision PS2
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Monday, October 5, 2015
## 
## Code synopsis: this file contains a python script to impelent a Hough Circle detector function
## I/O: N/A

import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import misc
import cv2
from detectCircles import detectCircles

image = cv2.imread("egg.jpg")

# input -> image to detect circles in, radius of circles to detect, and 0/1 (use gradient or not)
detectCircles(image,5,0)

