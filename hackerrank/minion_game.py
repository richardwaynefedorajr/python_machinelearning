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
