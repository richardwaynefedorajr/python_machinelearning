# Hacker rank coding challenges

Collection of python implementations for coding challenges located [here](https://www.hackerrank.com/domains/python?filters%5Bdifficulty%5D%5B%5D=medium&filters%5Bdifficulty%5D%5B%5D=hard)

Note:  these challenges are generally asking to read inputs from standard input in some specified format... this is annoying to debug because you have to enter data manually every time you run your program, so generally these implementations will have input values hard-coded... some include the functionality to read inputs as well.

## [Write a function](https://www.hackerrank.com/challenges/write-a-function/problem?isFullScreen=true)

## [The Minion Game](https://www.hackerrank.com/challenges/the-minion-game?isFullScreen=true)

## [Merge the Tools](https://www.hackerrank.com/challenges/merge-the-tools?isFullScreen=true)

## [Time Delta](https://www.hackerrank.com/challenges/python-time-delta?isFullScreen=true)

## [Find Angle ABC](https://www.hackerrank.com/challenges/find-angle/problem?isFullScreen=true)

## [No Idea!](https://www.hackerrank.com/challenges/no-idea/problem?isFullScreen=true)

## [Word Order](https://www.hackerrank.com/challenges/word-order/problem?isFullScreen=true)

## [Compress the String!](https://www.hackerrank.com/challenges/compress-the-string/problem?isFullScreen=true)

## [Company Logo](https://www.hackerrank.com/challenges/most-commons/problem?isFullScreen=true)

## [Piling Up!](https://www.hackerrank.com/challenges/piling-up/problem?isFullScreen=true)

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

## [Validating Email Adresses With a Filter](https://www.hackerrank.com/challenges/validate-list-of-email-address-with-filter/problem?isFullScreen=true)

## [Reduce Function](https://www.hackerrank.com/challenges/reduce-function/problem?isFullScreen=true)

## [Regex Substitution](https://www.hackerrank.com/challenges/re-sub-regex-substitution/problem?isFullScreen=true)

## [Validating Credit Card Numbers](https://www.hackerrank.com/challenges/validating-credit-card-number/problem?isFullScreen=true)

## [Words Score](https://www.hackerrank.com/challenges/words-score/problem?isFullScreen=true)

In this debugging problem, 
``` score += 1 ```
replaces
``` ++score ```
in
```
        if num_vowels % 2 == 0:
            score += 2
        else:
            ++score
```

## [Default Arguments](https://www.hackerrank.com/challenges/default-arguments/problem?isFullScreen=true)

Skipped for now

## [Maximize It!](https://www.hackerrank.com/challenges/maximize-it/problem?isFullScreen=true)

Note: hackerrank is not allowing numpy to be imported for this solution, but I preferred the cleanliness of numpy's outer sum, so I used that instead of a "by-hand" solution with itertools or similar.

## [Validating Postal Codes](https://www.hackerrank.com/challenges/validating-postalcode/problem?isFullScreen=true)

