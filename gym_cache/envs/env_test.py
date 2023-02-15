from gym_cache.envs.cache_env import CacheEnv
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

rew = np.loadtxt("/home/officer/mqr/Cache/maddpg_pytorch/models/20210401_data/model/run57/reward_data.txt")
print(rew)
x = []
for i in range(len(rew)):
  x.append(i)
fig = plt.figure()
# Plotting

plt.plot(rew, "*")


 # changes limits of x or y axis so that equal increments of x and y have the same length
plt.xlabel("x")
plt.ylabel("y")

plt.axis('equal')

plt.show()