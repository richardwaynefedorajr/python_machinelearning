## check if a month contains a Friday the 13th, given integer month and year

from datetime import date

month = 9
year = 2019
friday_13_result = 'IS' if date(year, month, 13).weekday() == 4 else 'is not'
print(str(month)+'/13/'+str(year)+' '+friday_13_result+' a Friday the 13th')