# Hacker rank coding challenges

Collection of python implementations for coding challenges located [here](https://www.hackerrank.com/domains/python?filters%5Bdifficulty%5D%5B%5D=medium&filters%5Bdifficulty%5D%5B%5D=hard)

Note:  these challenges are generally asking to read inputs from standard input in some specified format... this is annoying to debug because you have to enter data manually every time you run your program, so generally these implementations will have input values hard-coded... some include the functionality to read inputs as well.

## [Find angle](https://www.hackerrank.com/challenges/find-angle/problem?isFullScreen=true)

## [No Idea](https://www.hackerrank.com/challenges/no-idea/problem?isFullScreen=true)

## [Word Order](https://www.hackerrank.com/challenges/word-order/problem?isFullScreen=true)

## [Compress the String](https://www.hackerrank.com/challenges/compress-the-string/problem?isFullScreen=true)

## [Company Logo](https://www.hackerrank.com/challenges/most-commons/problem?isFullScreen=true)

## [Piling Up](https://www.hackerrank.com/challenges/piling-up/problem?isFullScreen=true)

## [Triangle Quest 2](https://www.hackerrank.com/challenges/triangle-quest-2/problem?isFullScreen=true)
Note: this is a math trick, not a programming trick... this implementation is splitting the difference, using reduce to leverage the fact that 1^2 = 1, 11^2 = 121, 111^2 = 12321, etc... but it will not pass on hackerrank, because the import adds an extra line.  To pass on hackerrank, you could use 
```
    print((10**(i)//9)**2)
```
for instance... indeed, even the usage of the trick implemented here was only for the purpose of removing spaces... otherwise, 
```
    print(*list(range(1,i+1))+list(range(i-1,0,-1)))
```
would do the trick, if spaces between integers were acceptable, or 
```
    print(*list(range(1,i+1))+list(range(i-1,0,-1)), sep='')
```
if hackerrank didn't fail it for string related inclusion of the 
```
sep=''
```

## [Itereables and Iterators](https://www.hackerrank.com/challenges/iterables-and-iterators/problem?isFullScreen=true)

## [Triangle Queset](https://www.hackerrank.com/challenges/python-quest-1/problem?isFullScreen=true)
Note: this is the same math trick as [Triangle Quest 2](https://www.hackerrank.com/challenges/triangle-quest-2/problem?isFullScreen=true) -> see above

## [Classes: Dealing with Complex Numbers](https://www.hackerrank.com/challenges/class-1-dealing-with-complex-numbers/problem?isFullScreen=true)
Note: solution c/p'd from hackerrank to read from standard input:
```
c = map(float, input().split())
d = map(float, input().split())
x = Complex(*c)
y = Complex(*d)
print(*map(str, [x+y, x-y, x*y, x/y, x.mod(), y.mod()]), sep='\n')
```

## [Athlete Sort](https://www.hackerrank.com/challenges/python-sort-sort/problem?isFullScreen=true)

## [ginortS](https://www.hackerrank.com/challenges/ginorts/problem?isFullScreen=true)

