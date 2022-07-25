import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import xlsxwriter
from lateralenv import *

n=10000
res= 0.1
lat=np.linspace(1,n,int(n/res)) 
long = 100* np.sin(np.radians(lat))
workbook = xlsxwriter.Workbook('sin road.xlsx')
road = workbook.add_worksheet("sin road")
road.write(0,0, "lat")
road.write(0,1, "long")
for i in range(1,len(lat)):
    road.write(i,0, lat[i])
    road.write(i,1, long[i])    
workbook.close()

# file = pd.read_excel("sin road.xlsx")
# road= file.to_numpy()
# # road= road[0:1000, :]
# lat=road[:,0]
# long=road[:,1]
# plt.xlim(0,200 )
# plt.ylim(-100, 100)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.draw()
# plt.plot(lat, long)
# x= np.linspace(0,200, 200)
# y=x
# plt.plot(x,y)
# plt.show()


# slope=[]
# for i in range(len(lat)-1):
#     slope.append((road[i+1, 1] - road[i, 1])/(road[i+1, 0] - road[i, 0]) )
#     dist, angle_diff, pre_point2 = lateralenv.dist_diff(ep=0, limit_dist=1, limit_ang=0, stp=0 ,pre_point=None)
# print(slope)
# lateralenv.dist_diff
# dist, angle_diff, pre_point2 = lateralenv.dist_diff(ep=0, limit_dist=1, limit_ang=0, stp=0 ,pre_point=None)

## for rho: -------------------------------------------------------

# file = pd.read_excel("sin road.xlsx")
# #lat/long data
# lat=file["lat"]
# long=file["long"]

# #transforming function
# def merc_from_arrays(lat, long):
#     r_major = 6378137.000
#     x = r_major * np.radians(long)
#     scale = x/long
#     y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + lat * (np.pi/180.0)/2.0)) * scale
#     return (x, y)

# #cartesian coordinate results
# xs, ys = merc_from_arrays(lat, long)
# fig=plt.plot(xs,ys)
# # plt.show()
# Rlist=[]
# signnumber=0
# r12=[(xs[1]-xs[0]),(ys[1]-ys[0])]
# r23=[(xs[2]-xs[1]),(ys[2]-ys[1])]
# s=np.cross(r12,r23)
# for i in range (0,620):
#     a=math.sqrt((xs[i+1]-xs[i])**2+(ys[i+1]-ys[i])**2)
#     b=math.sqrt((xs[i+2]-xs[i+1])**2+(ys[i+2]-ys[i+1])**2)
#     c=math.sqrt((xs[i+2]-xs[i])**2+(ys[i+2]-ys[i])**2)
#     q=(a**2+b**2-c**2)/(2*a*b)
#     R=c/(2*math.sqrt(1-q**2))
#     Rlist.append(R)
#     #sign alteration
#     r_a=[(xs[i+2]-xs[i+1]),(ys[i+2]-ys[i+1])]
#     r_b=[(xs[i+3]-xs[i+2]),(ys[i+3]-ys[i+2])]
#     w=np.cross(r_a,r_b)
#     sign=s*w
#     if(sign<0):
#         signnumber+=1
#     s=w
    
# print(1/max(Rlist), 1/min(Rlist))
    