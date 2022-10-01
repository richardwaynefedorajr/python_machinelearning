## File name: detectCircles.py
## Project specifics: (package, workspace, etc.) - ECE 5554 Computer Vision PS2
## Author: Richard Wayne Fedora Jr. 
## Email: rfedora1@vt.edu
## Phone: (706) 254-8887
## Date of most recent edit: Monday, October 5, 20.7
## 
## Code synopsis: this file contains a python function which implements the Hough Transform Circle detector
## I/O: 
## im - the image to be processed
## radius - specifies size of circle to be detected
## useGradient - option to exploit gradient direction measured at edge points
## centers - output matrix of [x,y] positions of detected circle centers

import numpy as np
from numpy import ndarray
from scipy import misc
from skimage import feature
from skimage import color
from skimage import draw
import matplotlib.pyplot as plt
import math

# add an if radius = 0 -> use variable radius
# add an if useGradient -> replace angle with gradient
# replace the np.where(...) for center locations with lists center_x, center_y, center_r


def detectCircles(im, radius, useGradient):
    gray = color.rgb2gray(im)
    if (radius == 0):
        H = np.zeros([gray.shape[0], gray.shape[1], round(0.05*len(gray[0,:]))])
    else:
        H = np.zeros(gray.shape)
    #theta = np.arange(0,math.pi*2, 0.1*math.pi)
    theta = np.arange(0,360,1)
    binary = feature.canny(gray, 1.5)
    edge_pixel = np.where(binary==1)
    gradient = np.gradient(gray)
    for i in range(len(edge_pixel[0])):
        if (radius == 0):
            for r in range(len(H[0,0,:])):
                if (useGradient == 0):
                    for angle in range(len(theta)):
                        a = edge_pixel[0][i] - 10*r*np.cos(angle)
                        b = edge_pixel[1][i] + 10*r*np.sin(angle)
                        if (round(a) < H.shape[0] and round(b) < H.shape[1]):
                            H[round(a),round(b), r] += 1
                else:
                    x_grad = gradient[0][edge_pixel[0][i],edge_pixel[1][i]]
                    y_grad = gradient[1][edge_pixel[0][i],edge_pixel[1][i]]
                    if (x_grad != 0):
                        angle = [np.rad2deg(np.arctan(y_grad/x_grad)), -np.rad2deg(np.arctan(y_grad/x_grad))]
                    else:
                        angle = []
                    for k in range(len(angle)):
                        a = edge_pixel[0][k] - 10*r*np.cos(angle[k])
                        b = edge_pixel[1][k] + 10*r*np.sin(angle[k])
                        if (round(a) < H.shape[0] and round(b) < H.shape[1]):
                            H[round(a),round(b),r] += 1

        else:
            if (useGradient == 0):
                for angle in range(len(theta)):
                    a = edge_pixel[0][i] - radius*np.cos(angle)
                    b = edge_pixel[1][i] + radius*np.sin(angle)
                    if (round(a) < H.shape[0] and round(b) < H.shape[1]):
                        H[round(a),round(b)] += 1
            else:
                x_grad = gradient[0][edge_pixel[0][i],edge_pixel[1][i]]
                y_grad = gradient[1][edge_pixel[0][i],edge_pixel[1][i]]
                if (x_grad != 0):
                    angle = [np.rad2deg(np.arctan(y_grad/x_grad)), np.rad2deg(np.arctan(y_grad/x_grad))-180]
                else:
                    angle = []
                for k in range(len(angle)):
                    a = edge_pixel[0][k] - radius*np.cos(angle[k])
                    b = edge_pixel[1][k] + radius*np.sin(angle[k])
                    if (round(a) < H.shape[0] and round(b) < H.shape[1]):
                        H[round(a),round(b)] += 1

    if (radius != 0):
        if (np.where(H > 0.7*np.max(H))[0].size == 1):
            [rr,cc] = draw.circle_perimeter(np.where(H > 0.7*np.max(H))[0], np.where(H > 0.7*np.max(H))[1], radius)
            if np.max(rr) < len(H[:,0]) and np.max(cc) < len(H[0,:]):
                im[rr,cc,0] = 255
                im[rr,cc,1] = im[rr,cc,2] = 0
        else:
            for i in range(len(np.where(H > 0.7*np.max(H))[0])):
                [rr,cc] = draw.circle_perimeter(np.where(H > 0.7*np.max(H))[0][i], np.where(H > 0.7*np.max(H))[1][i], radius)
                if np.max(rr) < len(H[:,0]) and np.max(cc) < len(H[0,:]):
                    im[rr,cc,0] = 255
                    im[rr,cc,1] = im[rr,cc,2] = 0
    else:
        if (np.where(H > 0.7*np.max(H))[0].size == 1):
            [rr,cc] = draw.circle_perimeter(np.where(H > 0.7*np.max(H))[0], np.where(H > 0.7*np.max(H))[1], 10*np.where(H > 0.7*np.max(H))[2])
            if np.max(rr) < len(H[:,0]) and np.max(cc) < len(H[0,:]):
                im[rr,cc,0] = 255
                im[rr,cc,1] = im[rr,cc,2] = 0
        else:
            for i in range(len(np.where(H > 0.7*np.max(H))[0])):
                [rr,cc] = draw.circle_perimeter(np.where(H > 0.7*np.max(H))[0][i], np.where(H > 0.7*np.max(H))[1][i], 10*np.where(H > 0.7*np.max(H))[2][i])
                if np.max(rr) < len(H[:,0]) and np.max(cc) < len(H[0,:]):
                    im[rr,cc,0] = 255
                    im[rr,cc,1] = im[rr,cc,2] = 0


    imgplot = plt.imshow(binary)
    plt.show()
    #misc.imsave('jupiter_edges_r_var_no_gradient.png', binary)
    if (radius == 0):
        for i in range(len(H[0,0,:])):
            imgplot = plt.imshow(H[:,:,i])
            plt.show()
            #misc.imsave('jupiter_H_r_30_no_gradient.png', H[:,:,i])
    else:
        imgplot = plt.imshow(H)
        plt.show()
        #misc.imsave('jupiter_H_r_30_gradient.png', H)
 
    imgplot = plt.imshow(im)
    plt.show()
    #misc.imsave('jupiter_r_30_gradient.png', im)

    # egg_edges_r_5_no_gradient
    # egg_H_r_var_no_gradient
    # egg_r_5_no_gradient
    # same for jupiter

    return 112
