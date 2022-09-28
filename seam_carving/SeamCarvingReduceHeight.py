## File name: SeamCarvingReduceHeight.py
## Project specifics: ECE 5554 Computer Vision - PS1
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Tuesday, Septermber 8, 2015
## 
## Code synopsis: the file contains a python script to execute seam carving to reduce the height of an image in a content-aware manner

import numpy as np
import cv2
from energy_image import energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_horizontal_seam import find_optimal_horizontal_seam
from displaySeam import displaySeam
from reduceHeight import reduceHeight
import matplotlib.pyplot as plt

im_in = cv2.imread('inputSeamCarvingPrague.jpg')
im = np.copy(im_in)
#im_resize = misc.imresize(im,[284,575])
#imisc.imsave('output_Prague_misc_resize.png',im_resize)
for i in range(100):
    energyImage = energy_image(im,'HORIZONTAL')
    energyImage += np.amax(energyImage)*0.5*im[:,:,2]
    img = plt.imshow(energyImage)
    img.set_cmap('gist_ncar')
    plt.show(img)
    cumulativeEnergyMap = cumulative_minimum_energy_map(energyImage,'HORIZONTAL')
    img2 = plt.imshow(cumulativeEnergyMap)
    img2.set_cmap('gist_ncar')
    plt.show(img2)
    horizontalSeam = find_optimal_horizontal_seam(cumulativeEnergyMap)
    seam_im = displaySeam(im,horizontalSeam,'HORIZONTAL')
    img3 = plt.imshow(seam_im)
    plt.show(img3)
    [im,energyImage] = reduceHeight(im,energyImage,horizontalSeam)

#misc.imsave('outputReducedHeightPrague.png',im)
