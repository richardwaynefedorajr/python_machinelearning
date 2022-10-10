from collections import Counter

s = 'qwertyuiopasdfghjklzxcvbnm'

## counter outputs dictionary of values and occurences
## sorted sorts dictionary, lambda dictates sorting by value, then key
## -x[1] means sorting in ascending order for negative value of occurences, equivalent to sorting values in reverse order
counts = sorted(Counter(s).items(), key=lambda x: (-x[1],x[0]), reverse=False)

for pair in counts[0:3]:
    print(pair[0], pair[1])