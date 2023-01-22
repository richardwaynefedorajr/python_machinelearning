# Computer Vision

Top level directory for various computer vision implementations

## Seam Carving

Content aware image re-sizing is accomplished by applying an energy function to an image, and further calculating the gradient of said
function either on the horizontal or vertical... optimal seam to remove can then be determined from gradient values.

Implementation of method presented in 

@proceedings{10.1145/1275808,
title = {SIGGRAPH '07: ACM SIGGRAPH 2007 Papers},
year = {2007},
isbn = {9781450378369},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
abstract = {In his classic 1835 study of American democracy, Alexis de Tocqueville defended the importance in free societies of associations and public meetings, observing that, "sentiments and ideas renew themselves, the heart is enlarged, and the human mind is developed only by the reciprocal action of men upon one another." In this spirit (except for de Tocqueville's gender bias), I welcome you to the 34th annual meeting of the Association for Computing Machinery's Special Interest Group on Graphics.The ACM SIGGRAPH Papers program is the premier international forum for disseminating new scholarly work in computer graphics. This year 455 papers were submitted, from which the Papers Committee accepted 108 papers - a new record. The acceptance rate was therefore 23.7%, the highest since 1984. These papers span the core areas of modeling, animation, rendering, and imaging, but they also touch on related areas such as visualization, computer vision, human-computer interaction, and applications of computer graphics. Since 2002 these proceedings have been published as a special issue of the journal ACM Transactions on Graphics.},
location = {San Diego, California}
}

| ![input_image](./test_images/inputSeamCarvingPrague.jpg) |
|:--:| 
| *Input image* |

| ![input_image_energy_map](./output_prague/prague_energy_map.png) |
|:--:| 
| *Energy map* |

| ![horizontal_gradient](./output_prague/prague_cmem_horiz.png) | ![vertical_gradient](./output_prague/prague_cmem_vertical.png) |
|:--:| 
| *Horizontal and vertical gradient images* |

| ![reduced_height](./output_prague/outputReducedHeightPrague.png) | ![reduced_width](./output_prague/outputReduceWidthPrague.png) |
|:--:| 
| *Reduced size output images* |


## Circle Detection

Canny edge detector generates a binary image from the input, and then applies Hough Transform to detect
the centers of circles with radius in provided range. 

| ![circle_detection_balloons output](./balloons_detected.png) |
|:--:| 
| *Circles detected in balloons test image* |

| ![circle_detection_egg output](./egg_detected.png) |
|:--:| 
| *Circles detected in egg test image* |