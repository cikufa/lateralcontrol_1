import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

discreate_road_pd = pd.read_csv('Chaos-generted road.csv')
discreate_road_np = discreate_road_pd.to_numpy()

x_acc = 0.5
new_road =[]

for i in range(discreate_road_np.shape[0]-1):
    new_road.append(discreate_road_np[i,:])

    point_number = int(abs((discreate_road_np[i+1,0] - discreate_road_np[i,0])//x_acc))
    print(point_number)

    diff = (discreate_road_np[i+1,0] - discreate_road_np[i,0])/point_number
    print(diff)
    x = discreate_road_np[i,0]
    y = discreate_road_np[i,1]
    for j in range(point_number-1):
        x += diff
        y += ((discreate_road_np[i+1,1]-discreate_road_np[i,1])/(discreate_road_np[i+1,0]-discreate_road_np[i,0]))*diff
        new_road.append([x,y])

new_road = np.array(new_road)
plt.plot(discreate_road_np[:,0], discreate_road_np[:,1])
plt.show()
plt.cla()
plt.plot(new_road[:,0], new_road[:,1])
plt.show()

pd.DataFrame(new_road).to_csv('acc05.csv')