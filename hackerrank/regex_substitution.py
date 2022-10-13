import re

lines = ['x&& &&& && && x || | ||\|| x']

# lines = []
# for _ in range(int(input())):
#     lines.append(str(input()))

lines = [re.sub("(?<=[\s])\&\&(?=[\s])", 'and', i) for i in lines]
lines = [re.sub("(?<=[\s])\|\|(?=[\s])", 'or', i) for i in lines]

print(*lines, sep="\n")