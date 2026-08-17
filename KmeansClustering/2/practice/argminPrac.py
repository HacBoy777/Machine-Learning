import numpy as np 


# my_arr = np.array([10,2,14,11,5])
# myindex= np.argmin(my_arr)
# print(myindex)

myarr = np.array([[10,50,30],[5,8,3],[14,35,78]])
# myindex = np.argmin(myarr,axis=1) ##[0 2 0]
myindex = np.argmin(myarr,axis=0)  ##[1 1 1] 
print(myindex)  