## There is an array of n integers. There are also 2 disjoint sets, A and B, each containing m integers. 
## You like all the integers in set A and dislike all the integers in set B. Your initial happiness is 0. 
## For each i integer in the array, if i is in A, you add 1 to your happiness. If i is in B, you add -1 to your happiness. 
## Otherwise, your happiness does not change. Output your final happiness at the end.

from functools import reduce
from operator import add

def incrementHappiness(A, B, val):
    if val in A:
        return 1
    elif val in B:
        return -1
    else:
        return 0
    
## to read from stdin as is described in the problem and necessary for code to run in hackerrank compiler
#n, m = map(int,input().split())
#int_array = list(map(int,input().split()))
#A = set(map(int,input().split()))
#B = set(map(int,input().split()))

n = 3
m = 2
int_array = [1, 5, 3]
A = {3, 1}
B = {5, 7}

happiness = reduce(add, map(lambda val: incrementHappiness(A, B, val), int_array))
print(happiness)
