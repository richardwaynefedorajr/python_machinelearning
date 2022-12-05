## Given two timestamps in format Day dd Mon yyyy hh:mm:ss +xxxx, print the absolute difference (in seconds) between them

from datetime import datetime

def time_delta(t1, t2):
    delta = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z') - datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    return str(int(abs(delta.total_seconds())))

t1 = "Sun 10 May 2015 13:54:36 -0700"
t2 = "Sun 10 May 2015 13:54:36 -0000"

print(time_delta(t1,t2))