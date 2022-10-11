from functools import reduce
from itertools import combinations
from math import factorial

## Enter your code here. Read input from STDIN. Print output to STDOUT
##N = int(input())
##letters = ''.join(str(input()).split())
##K = int(input())

letters = 'aacd'
K = 2
combos = factorial(len(letters)) / (factorial(len(letters) - K) * factorial(K))
print(reduce(lambda accumulator, subset: accumulator+1 if 'a' in subset else accumulator, combinations(letters,K), 0) / combos)