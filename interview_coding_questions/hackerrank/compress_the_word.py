from itertools import groupby
from functools import reduce

## get input from terminal or coded
##input_value = str(input())
input_value = '1222311'

## groupby iterator
input_iterator = groupby(input_value)

input_string_list = []

## list of values grouped by and mumber of occurences
for key, group in input_iterator:
    input_string_list += ['('+str(len(list(group)))+', '+str(key)+')']
    
## formatting (add spaces)
input_string = ' '.join(input_string_list)

print(input_string)