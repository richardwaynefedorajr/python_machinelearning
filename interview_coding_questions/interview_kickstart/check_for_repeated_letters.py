## Write a function in Python to return words with repeated letters from a sentence

import itertools as it
from collections import Counter

def checkForRepeatedLetters(word):
    letter_occurences = Counter(word).values()
    repeated_letters = it.filterfalse(lambda letters: letters < 2, letter_occurences)
    return True if any(repeated_letters) else False

sentence = 'This is a good sentence'
print('Repeated letters in sentence: \"'+sentence+'\":')    
print(list(filter(checkForRepeatedLetters, sentence.split())))