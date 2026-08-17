import numpy as np

home = [10,20,30,40,50]
jugnu = [1,2,3,4,5,6,7,8,9]

dist_arr = []

for h in home:  # 4 Times
    for j in jugnu:  # 10 Times
        dist = np.square(h-j)
        dist_arr.append(dist)

print(sorted(dist_arr))
        