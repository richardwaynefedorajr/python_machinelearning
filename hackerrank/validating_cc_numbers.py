import re
from itertools import filterfalse

card_numbers = ['5122-2368-7954-3214','4424424424442444','4424444424442444','acd']

# card_numbers = []
# for _ in range(int(input())):
#     card_numbers.append(str(input()))

r1 = re.compile(r'^[456][0-9]{3}-?[0-9]{4}-?[0-9]{4}-?[0-9]{4}$')
r2 = re.compile(r"([0-9])(-?\1){3}") #\1{4}
card_numbers = list(map(lambda x: 'Valid' if r1.search(x) and not r2.search(x) else 'Invalid', card_numbers))
print(*card_numbers, sep="\n")