import re

def fun(s):
    # return True if s is a valid email, else return False
    if s.count('@') == 1 and s.count('.') == 1:
        at_ind = s.find('@')
        dot_ind = s.find('.')
        if at_ind > dot_ind or dot_ind == at_ind + 1 or at_ind == 0 or dot_ind == len(s) - 1:
            return False
        else:
            user = s[0:at_ind]
            website = s[at_ind+1:dot_ind]
            ext = s[dot_ind+1:]
            
            if not re.match("^[A-Za-z0-9_-]*$", user):
                return False
            elif not website.isalnum():
                return False
            elif not ext.isalpha():
                return False
            elif len(ext) > 3:
                return False
            else:
                return True
    else:
        return False
    
def filter_mail(emails):
    return list(filter(fun, emails))
    
    
#print(filter_mail(['brian-23@hackerrank.com', 'britts_54@hackerrank.com', 'lara@hackerrank.com']))
print(filter_mail(['itsme@gmail','@something','@something.com','@something.co1','sone.com']))