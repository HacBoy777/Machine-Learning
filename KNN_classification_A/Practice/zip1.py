import numpy as np

a=["cheese","paneer","Dal"]
b=["Burgger","Roll","Rice"]
x=zip(a,b)
# print(list(x))
topings,base=zip(*x)
print("Topiings are : ",topings)
print("Base : ",base)