## File name: cumulative_minimum_energy_map.py
## Project specifics: ECE 5554 Computer Vision - PS1 
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Tuesday, September 8, 2015
## 
## Code synopsis: this file contains a python function to compute minimum cumulative energy.
##
## Inputs  - energyImage: a 2D matrix of datatype double (output of function energy_image)
##         - seamDirection: string 'HORIZONTAL' or 'VERTICAL'
## Outputs - cumulativeEnergyMap: 2D matrix of datatype double 

import numpy as np

def cumulative_minimum_energy_map(energyImage,seamDirection):
    M = np.copy(energyImage)
    if seamDirection == 'HORIZONTAL':
        for i in range(1,len(M[0,:])):
            for j in range(len(M[:,0])):
                if j == 0:
                    M[j,i] += min(M[j,i-1],M[j+1,i-1])
                elif j == len(M[:,0]) - 1:
                    M[j,i] += min(M[j-1,i-1],M[j,i-1])
                else:
                    M[j,i] += min(M[j-1,i-1],M[j,i-1],M[j+1,i-1])
    elif seamDirection == 'VERTICAL':
        for i in range(1,len(M[:,0])):
            for j in range(len(M[0,:])):
                if j == 0:
                    M[i,j] += min(M[i-1,j],M[i-1,j+1])
                elif j == len(M[0,:]) - 1:
                    M[i,j] += min(M[i-1,j-1],M[i-1,j])
                else:
                    M[i,j] += min(M[i-1,j-1],M[i-1,j],M[i-1,j+1])
    else:
        print('No seam direction given')
    
    return M
