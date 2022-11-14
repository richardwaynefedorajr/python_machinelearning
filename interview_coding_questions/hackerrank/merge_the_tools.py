from functools import reduce

def merge_the_tools(string, k):
    # indices for substrings
    for i in range(0,len(string),k):

        # accumulate substring on condition that next letter has not already been accumulated
        print(reduce(lambda a, b: a+b if b not in a else a, string[i:i+k]))

k = 3
input_string = "AABCAAADA"
merge_the_tools(input_string,k)

