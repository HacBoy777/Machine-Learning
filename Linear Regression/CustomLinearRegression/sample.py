import numpy as np


a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
a = np.array(a)
print(a)
print(a.shape)
a = a.reshape(15, 1)
a = a.reshape(a.shape[0],1)
print(a.shape)

for i in range(0,a.shape[0]):
    print(i,end = " ")
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
print("\n")
    
for i in range(0,a.shape[0],2):
    print(i,end = " ")
    # 0 2 4 6 8 10 12 14
print("\n")
    
for i in range(0,a.shape[0],3):
    print(i,end = " ")
    # 0 3 6 9 12
print("\n")