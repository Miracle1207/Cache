import numpy as np
import random

class DataRate(object):
    '''
    compute users' data rate when single transmission or joint transmission
    '''
    '''
    initial:
    users' position
    channel gain

    1.泊松分布 撒点初始化用户的位置
    2.获得坐标，计算用户距离edge server的距离d
    3.计算channel gain
    4.功率根据 edge server接入的用户量进行平分
    5.计算高斯噪声
    6.计算SINR
    7.计算data rate
    '''
    def __init__(self, user_n, edge_n):
        self.user_n = user_n
        self.edge_n = edge_n
        self.power_total = 19.953
        self.bandwidth = 4500000

    #def user_ppp(self,):
    def position_init(self):
        x = np.random.uniform(0, 1, self.user_n)
        y = np.random.uniform(0, 1, self.user_n)

    # Gaussian noise
    def compute_noise(self, NUM_Channel=1):
        ThermalNoisedBm = -174  # -174dBm/Hz
        var_noise = 10 ** ((ThermalNoisedBm - 30) / 10) * self.bandwidth / (
            NUM_Channel)  # envoriment noise is 1.9905e-15
        return var_noise

