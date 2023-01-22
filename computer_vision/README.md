# Computer Vision

Top level directory for various computer vision implementations

## Seam Carving

Content aware image re-sizing (Avidan and Shamir [2007](http://graphics.cs.cmu.edu/courses/15-463/2012_fall/hw/proj3-seamcarving/imret.pdf)) is accomplished by applying an energy function to an image, and further calculating the gradient of said
function either on the horizontal or vertical... optimal seam to remove can then be determined from gradient values.

| ![input_image](./seam_carving/test_images/inputSeamCarvingPrague.jpg) |
|:--:| 
| *Input image* |

| ![input_image_energy_map](./seam_carving/output_prague/prague_energy_map.png) |
|:--:| 
| *Energy map* |

| ![horizontal_gradient](./seam_carving/output_prague/prague_cmem_horiz.png) | ![vertical_gradient](./seam_carving/output_prague/prague_cmem_vertical.png) |
|:--:| 
| *Horizontal and vertical gradient images* |

| ![reduced_height](./seam_carving/output_prague/outputReducedHeightPrague.png) | ![reduced_width](./seam_carving/output_prague/outputReduceWidthPrague.png) |
|:--:| 
| *Reduced size output images* |


## Circle Detection

Canny edge detector generates a binary image from the input, and then applies Hough Transform to detect
the centers of circles with radius in provided range. 

| ![circle_detection_balloons output](./circle_detection/balloons_detected.png) |
|:--:| 
| *Circles detected in balloons test image* |

| ![circle_detection_egg output](./circle_detection/egg_detected.png) |
|:--:| 
| *Circles detected in egg test image* |