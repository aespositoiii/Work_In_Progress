#this functions creates a list of a number of prime numbers and prints them

def findnprimes(a):
    if type(a) != int:
        return 'NOOOOO! INTEGERS ONLY!'
    y = []
    i = 1
    while i <= (3 or a):
        y.append(i)
        i+=1
    while len(y) < a:
        isprime = 1
        for j in range(2,i):
            
            b = i % j
            isprime = isprime * b  
        if isprime != 0:
            
            y.append(i)
        i+=1
    print(y)

#this function creates a list containing the prime factorization of the number a
def primefactors(a):
    if type(a) != int:
        return 'NOOOOO! INTEGERS ONLY!'
    x = []
    y = [1]
    reduced = a
    rem = 0
    while rem == 0:
        for i in range(2,reduced):
            rem = reduced % i
            if rem == 0:
                reduced = reduced//i
                y.append(i)
                break
    if reduced == a:
        print("IS PRIME")
    y.append(reduced)
    print(y)
                
            
    