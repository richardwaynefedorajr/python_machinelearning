## Write a code in Python to create a Morse code translator.
## The code should return the Morse code that is equivalent to the string.
## Space between words is denoted with an _

alphabet = { 'A':'.-', 'B':'-...', 'C':'-.-.', 'D':'-..', 'E':'.',
             'F':'..-.', 'G':'--.', 'H':'....', 'I':'..', 'J':'.---', 'K':'-.-',
             'L':'.-..', 'M':'--', 'N':'-.', 'O':'---', 'P':'.--.', 'Q':'--.-',
             'R':'.-.', 'S':'...', 'T':'-', 'U':'..-', 'V':'...-', 'W':'.--',
             'X':'-..-', 'Y':'-.--', 'Z':'--..',
             '1':'.----', '2':'..---', '3':'...--', '4':'....-', '5':'.....', 
             '6':'-....', '7':'--...', '8':'---..', '9':'----.', '0':'-----', 
             ', ':'--..--', '.':'.-.-.-', '?':'..--..', '/':'-..-.', 
             '-':'-....-', '(':'-.--.', ')':'-.--.-', 
             ' ':'_' }

morse_code_sentence = 'This is a string'
print('Morse code translation of sentence: \"'+morse_code_sentence+'\":')
print(''.join(list(map(lambda character: alphabet[character], morse_code_sentence.upper()))))