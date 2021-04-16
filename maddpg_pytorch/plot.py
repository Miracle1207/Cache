import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import spline
from scipy.signal import savgol_filter
from matplotlib.patches import Circle
import pandas as pd


st_4000 = np.loadtxt("/home/mqr/Cache/maddpg_pytorch/0411-delay/MARL-ST.txt")
rl_4000 = np.loadtxt("/home/mqr/Cache/maddpg_pytorch/0411-delay/MARL-JT.txt")
bi_4000 = np.loadtxt("/home/mqr/Cache/maddpg_pytorch/0411-delay/0412-bi-level.txt")

lfu_4000 = np.loadtxt("/home/mqr/Cache/maddpg_pytorch/0411-delay/LFU-delay_1.txt")
fifo_4000 = np.loadtxt("/home/mqr/Cache/maddpg_pytorch/0411-delay/FIFO-delay.txt")
lru_4000 = np.loadtxt("/home/mqr/Cache/maddpg_pytorch/0411-delay/LRU-delay.txt")

plot_num = 4000
rl = rl_4000[:plot_num]
bi = bi_4000[:plot_num]
lfu = lfu_4000[:plot_num]
fifo = fifo_4000[:plot_num]
st = st_4000[:plot_num]
lru = lru_4000[:plot_num]
x = range(plot_num)
#
def smooth(x, y, color, label):
    window_size = 71
    polynomial_order = 3
    # 换算单位
    # y = y/1e6
    yhat = savgol_filter(y, window_size, polynomial_order)
    y_ = savgol_filter(y, 5, polynomial_order)
    plt.plot(x, y_, color=color, alpha=0.1)
    plt.plot(x, yhat, color=color, label=label)


smooth(x, st, "orange", "MARL - ST")
smooth(x, rl, "royalblue", "MARL - JT")
smooth(x, bi, "forestgreen", "MARL - BLA")
plt.ylim(3, 25)
# smooth(x, lfu, "darkorange", "LFU - JT")
# smooth(x, fifo, "forestgreen", "FIFO - JT")
# smooth(x, lru, "deepskyblue", "LRU - JT")
# smooth(x, rl, "royalblue", "MARL - JT")
plt.xlim(0, 4000)
# 设置x,y轴代表意思
plt.xlabel("Iterations")
plt.ylabel("Transmission Delay (s)")
# 设置标题
# plt.title("")
plt.legend()
plt.savefig('/home/mqr/Cache/maddpg_pytorch/pics/bi-cache.pdf')
#plt.savefig('/home/mqr/Cache/maddpg_pytorch/pics/RL-cache.pdf')
plt.show()



#
# '''plot users and edge servers'''
# user_x = np.loadtxt("/home/mqr/Cache/gym_cache/envs/user_ppp/1221/ex.txt")
# user_y = np.loadtxt("/home/mqr/Cache/gym_cache/envs/user_ppp/1221/ey.txt")
#
# # user_x = 100*user_x
# # user_y = 100 * user_y
#
# # np.savetxt("/home/mqr/Cache/gym_cache/envs/user_ppp/1221/ex.txt", user_x)
# # np.savetxt("/home/mqr/Cache/gym_cache/envs/user_ppp/1221/ey.txt", user_y)
# '''plot'''
# fig = plt.figure()
# # Plotting
# plt.scatter(user_x, user_y, edgecolor='b', facecolor='none', alpha=0.5, label="Users")
#
# # margin
#
# ax = fig.add_subplot(111)
# cir1 = Circle(xy=(0, 0), radius=100, alpha=0.1, color="b", label="E1 Coverage")
# cir2 = Circle(xy=(100, 0), radius=100, alpha=0.1, color="g", label="E2 Coverage")
# cir3 = Circle(xy=(50, 100), radius=100, alpha=0.1, color="r", label="E3 Coverage")
# ax.add_patch(cir1)
# ax.add_patch(cir2)
# ax.add_patch(cir3)
#
# ax.plot(0, 0, 'b*', label="Edge 1")
# ax.plot(100, 0, 'g*', label="Edge 2")
# ax.plot(50, 100, 'r*', label="Edge 3")
#
# plt.axis('scaled')
# plt.axis('equal')  # changes limits of x or y axis so that equal increments of x and y have the same length
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.axis('equal')
# plt.legend()
# plt.xlim(-125, 300)
# plt.savefig('/home/mqr/Cache/maddpg_pytorch/pics/users.pdf')
# plt.show()

