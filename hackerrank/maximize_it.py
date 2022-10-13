import numpy as np
from functools import reduce

arrays = []
N, M = input().split()
for _ in range(int(N)):
    arrays.append(np.square(np.array(list(map(int,input().split()))[1::])))   

result = reduce(lambda acc, arr: np.add.outer(acc, arr), arrays)
result = result % int(M)
print(result.max())