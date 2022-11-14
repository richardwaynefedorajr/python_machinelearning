from math import sin, asin, cos, atan, sqrt, degrees

##AB = int(input())
##BC = int(input())

AB = 10
BC = 10

## right triangle tan rule
angle_BCM = atan(AB/BC)

## pythagorean theorem
MC = 0.5*sqrt(AB**2 + BC**2)

## law of cosines
MB = sqrt(BC**2 + MC**2 - 2*BC*MC*cos(angle_BCM))

## law of sines
theta = degrees( asin( sin(angle_BCM)*MC / MB ) )
print(str(round(theta))+chr(176))
