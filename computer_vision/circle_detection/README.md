# Execution instructions

Main interface for execution is hough_implementation.py: 

```
# read image
#image = cv2.imread("egg.jpg")
image = cv2.imread("balloons.jpg")

# detect circles
detectCircles(image, 20, 40, 0.5)
```

detectCircles contains the hough detector implementation:

```
# im -> input image
# r_min, r_max -> min and max radius values of circles to detect
# threshold -> percentage of max vote getter in accumulator to consider as detected circles
#               i.e. any pixel receiving threshold*max_accumulated_value votes will be considered a detected circle

def detectCircles(im, r_min, r_max, threshold):
```

This implementation of the Hough Transform Circle Detector takes an input
image, uses a Canny edge detector to generate a binary image from the input
image converted to grayscale, and then uses the Hough Transform to detect
the centers of circles with radius in provided range. First, the indices of all
edge pixels are extracted, and for each edge pixel, a circle is generated in the
Hough space with a discretized set of angles and of either fixed radius or over
the range of expected radii, where each edge pixel casts a vote for the center
location. When the vote accumulator is fully generated (i.e. all edge pixels
have voted), the maximum vote getters are returned as the center locations of
the detected circles.
