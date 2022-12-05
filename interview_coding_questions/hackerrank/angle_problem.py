## ABC is a right triangle, 90 deg at B
## Therefore, ABC = 90 deg
## Point M is the midpoint of hypotenuse AC
## You are given the lengths AB and BC
## Your task is to find angle MBC in degrees.

from math import sin, asin, cos, atan, sqrt, degrees

## initialize known variables
#AB = int(input())
#BC = int(input())
AB = 10
BC = 10

## right triangle tan rule
angle_BCM = atan(AB/BC)

## pythagorean theorem
MC = 0.5*sqrt(AB**2 + BC**2)

## law of cosines
MB = sqrt(BC**2 + MC**2 - 2*BC*MC*cos(angle_BCM))

## law of sines
angle_MBC = degrees( asin( sin(angle_BCM)*MC / MB ) )

## print result
print(str(round(angle_MBC))+chr(176))
