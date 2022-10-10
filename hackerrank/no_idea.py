from functools import reduce

## to read from stdin as is described in the problem and necessary for code to run in hackerrank compiler
##n, m = map(int,input().split())
##int_array = list(map(int,input().split()))
##A = set(map(int,input().split()))
##B = set(map(int,input().split()))

n = 3
m = 2
int_array = [1, 5, 3]
A = {3, 1}
B = {5, 7}

happiness = reduce(lambda acc, val: acc+1 if val in A else (acc-1 if val in B else acc) , int_array, 0)
print(happiness)
