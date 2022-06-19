import numpy as np
x = [2]
y = [3]
minus = np.array([[1,1], [2,6], [4,7]])
z = minus - np.array((x,y)).reshape([1,2])
m = np.sqrt(z[:,0]**2 + z[:,1]**2)
print(z)
print(m)
print(np.min(m))
print(np.argmin(m))