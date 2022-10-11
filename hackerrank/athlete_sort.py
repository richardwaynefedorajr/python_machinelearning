import math
import os
import random
import re
import sys
from functools import reduce

##5 3
##10 2 5
##7 1 0
##9 9 9
##1 23 12
##6 5 9
##1

if __name__ == '__main__':
    # nm = input().split()

    # n = int(nm[0])

    # m = int(nm[1])

    # arr = []

    # for _ in range(n):
    #     arr.append(list(map(int, input().rstrip().split())))

    # k = int(input())

    arr = [[10, 2, 5], [7, 10, 0], [9, 9, 9], [1, 23, 12], [6, 5, 9]]
    k = 1
    print(*map(lambda x: ' '.join(map(str,x)), sorted(arr,key=lambda x: x[k])), sep="\n")