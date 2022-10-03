import numpy as np
from numpy import ndarray
from scipy import misc
from skimage import feature
from skimage import color
from skimage import draw
import matplotlib.pyplot as plt
import cv2 as cv
import math

# im -> input image
# r_min, r_max -> min and max radius values of circles to detect
# threshold -> percentage of max vote getter in accumulator to consider as detected circles
#               i.e. any pixel receiving threshold*max_accumulated_value votes will be considered a detected circle

def detectCircles(im, r_min, r_max, threshold):
    
    # canny edge detector
    gray = color.rgb2gray(im)
    edge_pixel = np.where(feature.canny(gray, 1.5) == 1)
    imgplot = plt.imshow(feature.canny(gray, 1.5))
    plt.show()

    # initialize accumulator matrix and angles/radii
    theta = np.arange(0,360,1)
    radius = np.arange(r_min, r_max, 1)
    H = np.zeros([gray.shape[0], gray.shape[1], len(radius)])

    # votes in hough space per each radius/angle combo
    a = np.multiply.outer(radius, np.cos(theta)).flatten()
    b = np.multiply.outer(radius, np.sin(theta)).flatten()
    r = np.multiply.outer(radius, np.ones(len(theta))).flatten()

    # apply votes to each edge coordinate
    x = np.round(np.subtract.outer(edge_pixel[0], a)).flatten()
    y = np.round(np.add.outer(edge_pixel[1], b)).flatten()
    r = np.multiply.outer(np.ones(len(edge_pixel[0])), r).flatten()

    # shift radius to index in accumulator, weed out indices that are out of bounds
    indices_in_bounds = np.where((x < H.shape[0]) & (x >= 0) & (y < H.shape[1]) & (y >= 0))[0]
    r -= r_min

    # accumulator for votes
    np.add.at(H,(x[indices_in_bounds].astype(int),y[indices_in_bounds].astype(int),r[indices_in_bounds].astype(int)),1)

    # get center indices and radii (re-shift radii indices to value)
    circles = np.asarray(np.where(H > threshold*np.max(H))).T
    circles[:,2] += r_min
    
    # print results and display
    print("Circles:")
    print(circles)
    for i in range(len(circles[:,0])):
        cv.circle(im,(circles[i,1],circles[i,0]),circles[i,2],(0,255,0),1)
    cv.namedWindow('image',cv.WINDOW_NORMAL)
    cv.resizeWindow('image', 600,600)
    cv.imshow('image',im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return 0
