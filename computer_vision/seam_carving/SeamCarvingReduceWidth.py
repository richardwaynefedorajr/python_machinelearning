## File name: SeamCarvingReduceWidth.py
## Project specifics: ECE 5554 Computer Vision - PS1
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Tuesday, Septermber 8, 2015
## 
## Code synopsis: the file contains a python script to execute seam carving to reduce the width of an image in a content-aware manner

import numpy as np
from scipy import misc
from energy_image import energy_image
from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_vertical_seam import find_optimal_vertical_seam
from displaySeam import displaySeam
from reduceWidth import reduceWidth
import matplotlib.pyplot as plt

im_in = misc.imread('inputSeamCarvingPrague.jpg')
im = np.copy(im_in)
#im_resize = misc.imresize(im,[360,480])
#misc.imsave('outputPrague_misc_resize.png',im_resize)
for i in range(100):
    energyImage = energy_image(im,'VERTICAL')
    #img = plt.imshow(energyImage,)
    #img.set_cmap('gist_ncar')
    #plt.show(img)
    cumulativeEnergyMap = cumulative_minimum_energy_map(energyImage,'VERTICAL')
    #img2 = plt.imshow(cumulativeEnergyMap)
    #img2.set_cmap('gist_ncar')
    #plt.show(img2)
    verticalSeam = find_optimal_vertical_seam(cumulativeEnergyMap)
    #seam_im = displaySeam(im,verticalSeam,'VERTICAL')
    #img3 = plt.imshow(seam_im)
    #plt.show(img3)
    [im,energyImage] = reduceWidth(im,energyImage,verticalSeam)

misc.imsave('outputReduceWidthPrague.png',im)
