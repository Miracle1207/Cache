import numpy as np
import random
import matplotlib.pyplot as plt

'''
initial:
user number
bandwidth B
channel gain h
power p
noise sigma
'''
user_num = 100
bandwidth = 4500000  # 4.5MHZ
p_total = 19.953  # total power is 19.953w, namely 43dbm
p = p_total/user_num

def channel_gain(user_n):
    h = abs(1 / np.sqrt(2) * (np.random.randn(user_n) + 1j * np.random.randn(user_n)))
    return h

# Gaussian noise
def compute_noise(NUM_Channel):
    ThermalNoisedBm = -174  # -174dBm/Hz
    var_noise = 10 ** ((ThermalNoisedBm - 30) / 10) * bandwidth / (
        NUM_Channel)  # envoriment noise is 1.9905e-15
    return var_noise

def R_compute(num_channel, h_gain):
    sum_p = 0
    for i in range(num_channel):
        sum_p += np.power(h_gain[i], 2)*p
    sinr = np.zeros(num_channel)
    rate = np.zeros(num_channel)
    for i in range(num_channel):
        sinr[i] = h_gain[i]*h_gain[i]*p/(sum_p-h_gain[i]*h_gain[i]*p + np.power(compute_noise(1), 2))
        rate[i] = bandwidth * np.log2(1+sinr[i])
    return np.mean(rate), np.sum(rate)

color = ['r','k','y','g','c','b','m']
def draw(index):
    r_mean = np.zeros(user_num)
    r_sum = np.zeros(user_num)
    for i in range(1, 100):
        h = channel_gain(i)
        r_mean[i], r_sum[i] = R_compute(i, h)
        print("date rate", i)
        print(r_mean[i])
        print(r_sum[i])
    user = range(1, user_num + 1)
    plt.plot(user, r_mean, '.', color=color[index], label='r_mean')
    plt.plot(user, r_sum, color=color[index], label='r_sum')

for k in range(7):
    draw(k)

plt.xlabel("user_number")
plt.ylabel("data rate")
plt.legend(loc='upper right')
plt.show()
# for i in range(1, 100):
#     print(compute_noise(i))
