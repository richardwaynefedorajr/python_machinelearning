##Kevin and Stuart want to play the 'The Minion Game'.

##Game Rules

##Both players are given the same string, .
##Both players have to make substrings using the letters of the string .
##Stuart has to make words starting with consonants.
##Kevin has to make words starting with vowels.
##The game ends when both players have made all possible substrings.

##Scoring
##A player gets +1 point for each occurrence of the substring in the string .

## Output string: the winner's name and score, separated by a space on one line, or Draw if there is no winner

from itertools import combinations
  
def getSubStrings(input_string):  
    input_string = input_string.lower()
    return set([input_string[x:y] for x, y in combinations(range(len(input_string) + 1), r = 2)])

initial_string = "banana"
substring_set = getSubStrings(initial_string)

kevin = 0
stuart = 0

for sub in substring_set:
    if sub[0] in 'aeiou':
        kevin += initial_string.count(sub)
    else:
        stuart += initial_string.count(sub)

if kevin > stuart:
    print("Kevin "+str(kevin))
elif stuart > kevin:
    print("Stuart "+str(stuart))
else:
    print("Draw")
