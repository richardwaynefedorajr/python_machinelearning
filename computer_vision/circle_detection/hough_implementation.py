import cv2
from detectCircles import detectCircles

# read image
#image = cv2.imread("egg.jpg")
image = cv2.imread("test_images/egg.jpg")

# detect circles
detectCircles(image, 4, 12, 0.5)