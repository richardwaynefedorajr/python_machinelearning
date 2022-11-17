## File name: find_optimal_horizontal_seam.py
## Project specifics: ECE 5554 Computer Vision - PS1
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Tuesday, September 8, 2015
## 
## Code synopsis: this file contains a python function to compute the optimal horizontal seam in an image
##
## Inputs  - cumulativeEnergyMap: 2D matrix of datatype double (output of cumulative_minimum_energy_map)
## Outputs - horizontalSeam: vector containing the column horizontalSeam of the pixels which form the seam for each column

import numpy as np

def find_optimal_horizontal_seam(cEM):
    horizontalSeam = []
    column = len(cEM[0,:]) - 1
    rows = len(cEM[:,0]) - 1
    horizontalSeam.insert(0,np.argmin(cEM[:,column]))
    column -= 1
    while column >= 0:
        if horizontalSeam[0] == 0:
            horizontalSeam.insert(0,np.argmin(cEM[0:2,column]))
        elif horizontalSeam[0] == rows:
            horizontalSeam.insert(0,rows+np.argmin(cEM[rows-1:rows+1,column])-1)
        else:
            horizontalSeam.insert(0,(horizontalSeam[0]-1)+np.argmin(cEM[(horizontalSeam[0]-1):(horizontalSeam[0]+2),column]))
        column -= 1
   
    return horizontalSeam 
