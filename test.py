import numpy as np

'''Done= 0
for i in range(2):
    print(i)
    if (i==0):
        Done=1
    while True:
        if Done==1 :
            break 
        print(6)
    print(8)'''

x = np.arange(1,95)
y = x**2

for i in range(0,x.shape[0],10):
    print(x[i:i+10])