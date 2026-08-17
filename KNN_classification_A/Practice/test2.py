import numpy as np

home = np.array([[10,20,30,40]])
jugnu = np.array([[1,2,3,4]])
# print(home.shape)  ## (1,4)
# print(jugnu.shape)  ## (1,4)

print(home-jugnu)  ## (1,4)  [[9,18,27,36]]
print(np.square(home-jugnu))  ## (1,4)  [[81,324,729,1296]]
print(np.sum(np.square(home-jugnu)))  ## (1,)  [2430]