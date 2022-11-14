from functools import reduce

words = {}
n = int(input())
for i in range(n):
    word = input()
    if word in words:
        words[word] += 1
    else:
        words[word] = 1
    
print(len(words))
print(' '.join(map(str,list(words.values()))))
