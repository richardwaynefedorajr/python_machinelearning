# Enter your code here. Read input from STDIN. Print output to STDOUT
def checkIfIntAndOdd(s):
    if s.isnumeric():
        return 1 if int(s) % 2 == 0 else 0
    else:
        return 0
    
string_in = input()
print(''.join(sorted(string_in, key=lambda s: (s.isnumeric(), s.isupper(), checkIfIntAndOdd(s), s))))