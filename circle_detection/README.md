# Execution instructions

Main interface for execution is hough_implementation.py: update image to read, radius, and whether or not to use gradient (0/1):

```
image = cv2.imread("egg.jpg")

# input -> image to detect circles in, radius of circles to detect, and 0/1 (use gradient or not)
detectCircles(image,5,0)
```
