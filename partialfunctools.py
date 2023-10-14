from functools import partial
# original function
def power(exponent, base):
    return base ** exponent
#Partial function   
square = partial(power, 2) # setting value of exponent to 2
cube = partial(power, 3) # setting value of exponent to 3
# Calling Partial function
print("The square of 5 is", square(5))
print("The cube of 7 is", cube(7))

print(int("101",base=10))