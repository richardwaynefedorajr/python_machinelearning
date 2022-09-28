## File name: find_optimal_vertical_seam.py
## Project specifics: ECE 5554 Computer Vision - PS1
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Tuesday, September 8, 2015
## 
## Code synopsis: this file contains a python function to compute the optimal vertical seam in an image
##
## Inputs  - cEM: a 2D matrix of datatype double (output of cumulative_minimum_energy_map)
## Outputs - verticalSeam: vector of column verticalSeam of the pixels which form the seam for each row

import numpy as np

def find_optimal_vertical_seam(cEM):
    verticalSeam = []
    row = len(cEM[:,0]) - 1
    columns = len(cEM[0,:]) - 1
    verticalSeam.insert(0,np.argmin(cEM[row,:]))
    row -= 1
    while row >= 0:
        if verticalSeam[0] == 0:
            verticalSeam.insert(0,np.argmin(cEM[row,0:2]))
        elif verticalSeam[0] == columns:
            verticalSeam.insert(0,columns+np.argmin(cEM[row,columns-2:columns])-1)
        else:
            verticalSeam.insert(0,verticalSeam[0]-1+np.argmin(cEM[row,verticalSeam[0]-1:verticalSeam[0]+2]))
        row -= 1

    return verticalSeam 
