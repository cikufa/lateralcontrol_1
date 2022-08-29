import numpy as np
import matplotlib.pyplot as plt
from main import *
import pandas as pd
discreate_road_pd = pd.read_excel('sin road.xlsx')
road = discreate_road_pd.to_numpy()
plt.plot(road[0:1000, 0],road[0:1000, 1])
plt.show()