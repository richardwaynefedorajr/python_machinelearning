from functools import reduce

#for i in range(1,int(input())+1): #More than 2 lines will result in 0 score. Do not leave a blank line also
N = 5
for i in range(1,N+1):
#    print(reduce(lambda acc, val: acc + 10**val, range(0,i), 0)**2)
    print(*list(range(1,i+1))+list(range(i-1,0,-1)), sep='')