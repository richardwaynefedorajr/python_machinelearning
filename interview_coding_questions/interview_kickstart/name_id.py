## Write a function in Python to parse a string such that it accepts a parameter- an encoded string. 
## This encoded string will contain a first name, last name, and an id. 
## You can separate the values in the string by any number of zeros. 
## The id will not contain any zeros. 
## The function should return a Python dictionary with the first name, last name, and id values. 
## For example, if the input would be "John000Doe000123". 
## Then the function should return: { "first_name": "John", "last_name": "Doe", "id": "123" }

import re

def getNameAndIdDict(input_string):
    split_string = re.split('0+',input_string)
    return {'first_name':split_string[0], 'last_name':split_string[1], 'id':split_string[2]}

encoded_string = 'John000Doe0123'
print('Parse encoded string: '+encoded_string)
print(getNameAndIdDict(encoded_string))