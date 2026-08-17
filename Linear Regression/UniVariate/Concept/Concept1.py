import numpy as np
list = np.array([10, 20, 30, 40, 50, 60])
# print(list.shape)
# print(list)

x = list[np.newaxis,:]
print(x.shape)
print(x)

y = list[:, np.newaxis]
print(y.shape)
print(y)