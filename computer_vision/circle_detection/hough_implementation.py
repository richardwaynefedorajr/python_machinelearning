import cv2
from detectCircles import detectCircles

# read image
#image = cv2.imread("egg.jpg")
image = cv2.imread("balloons.jpg")

# detect circles
detectCircles(image, 20, 40, 0.5)

