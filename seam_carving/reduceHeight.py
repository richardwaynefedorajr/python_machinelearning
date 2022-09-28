## File name: reduceHeight.py
## Project specifics: ECE 5554 Computer Vision - PS1
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Tuesday, Septermber 8, 2015
## 
## Code synopsis: this file contains a python function to reduce the height of an image and its corresponding minimum energy map by one pixel by removing pixels at seam indices
##
## Inputs  - im: MxNx3 image of datatype uint8
##         - energyImage: 2D matrix of datatype double (output of energy_image)
## Outputs - reducedColorImage: 3D matrix of reduced height, datatype uint8
##         - reducedEnergyImage: 2D matrix of reduced height, datatype double

import numpy as np

def reduceHeight(im, energyImage, indices):
    im_copy = np.copy(im)
    im_shape = im_copy.shape
    n_rows = im_shape[0]
    n_cols = im_shape[1]
    n_slices = im_shape[2]
    eI_copy = np.copy(energyImage)
    eI_shape = eI_copy.shape
    n_rows_ei = eI_shape[0]
    n_cols_ei = eI_shape[1]
    
    #index = np.zeros((n_cols,2))
    #for i in range(n_cols):
        #index[i,:] = [i,indices[i]]
        #index[i,:,1] = [i,indices[i]]
        #index[i,:,2] = [i,indices[i]]

    R_p = np.copy(im[:,:,0])
    G_p = np.copy(im[:,:,1])
    B_p = np.copy(im[:,:,2])
    reducedColorImage = np.zeros((n_rows-1,n_cols,n_slices),dtype=np.uint8)
    reducedEnergyImage = np.zeros((n_rows_ei-1,n_cols_ei))
    
    for i in range(n_cols):
        
        placeholder = np.copy(R_p[:,i])
        placeholder = np.delete(placeholder,indices[i])
        reducedColorImage[:,i,0] = np.copy(placeholder)
        
        placeholder = np.copy(G_p[:,i])
        placeholder = np.delete(placeholder,indices[i])
        reducedColorImage[:,i,1] = np.copy(placeholder)

        placeholder = np.copy(B_p[:,i])
        placeholder = np.delete(placeholder,indices[i])
        reducedColorImage[:,i,2] = np.copy(placeholder)

        placeholder = np.copy(energyImage[:,i])
        placeholder = np.delete(placeholder,indices[i])
        reducedEnergyImage[:,i] = np.copy(placeholder)

    return [reducedColorImage,reducedEnergyImage]
