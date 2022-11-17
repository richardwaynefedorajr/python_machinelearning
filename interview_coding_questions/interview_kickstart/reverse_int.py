## Write a function to reverse an integer

def reverseInt(input_int):
    return int(str(input_int)[::-1])

int_to_reverse = 2345
print('Reverse int '+str(int_to_reverse))
print(reverseInt(int_to_reverse))