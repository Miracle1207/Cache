import numpy as np
import random
import math
import copy

class DataRate(object):
    '''
    compute users' data rate when single transmission or joint transmission
    '''
    '''
    initial:

    channel gain

    1.泊松分布 撒点初始化用户的位置
    2.获得坐标，计算用户距离edge server的距离d
    3.计算channel gain
    4.功率根据 edge server接入的用户量进行平分
    5.计算高斯噪声
    6.计算SINR
    7.计算data rate
    '''
    def __init__(self, user_n=6, edge_n=2):
        self.user_n = user_n
        self.edge_n = edge_n
        self.power_total = 19.953
        self.bandwidth = 4500000
        self.r = 1
        self.xy = np.array([[0,0],[1,0]])

    #def user_ppp(self,):
    def position_init(self, xy, user_n):
        user_n = int(user_n)
        theta = 2 * np.pi * np.random.uniform(0, 1, user_n)  # angular coordinates
        rho = self.r * np.sqrt(np.random.uniform(0, 1, user_n))  # radial coordinates
        # Convert from polar to Cartesian coordinates
        xx = rho * np.cos(theta)
        yy = rho * np.sin(theta)
        # Shift centre of disk to (xx0,yy0)
        xx = xx + xy[0]
        yy = yy + xy[1]
        return xx, yy

    def distance_compute(self, ex, ey):
        d = np.zeros(shape=(self.edge_n, self.user_n))
        for i in range(self.edge_n):
            for j in range(self.user_n):
                d[i][j] = math.sqrt((ex[j]-self.xy[i][0])**2+(ey[j]-self.xy[i][1])**2)
        return d


    # Gaussian noise
    def compute_noise(self, NUM_Channel=1):
        ThermalNoisedBm = -174  # -174dBm/Hz
        var_noise = 10 ** ((ThermalNoisedBm - 30) / 10) * self.bandwidth / (
            NUM_Channel)  # envoriment noise is 1.9905e-15
        return var_noise

    def channel_gain(self, alpha, ex, ey):
        g = np.zeros(shape=(self.edge_n, self.user_n))
        d = self.distance_compute(ex, ey)
        h = np.zeros(shape=(self.edge_n, self.user_n))
        for i in range(self.edge_n):
            g[i] = abs(1 / np.sqrt(2) * (np.random.randn(self.user_n) + 1j * np.random.randn(self.user_n)))
            for j in range(self.user_n):
                h[i][j] = g[i][j]*(d[i][j]**(-alpha/2))
        return h
    '''
    connect_flag = (edge_n, user_n)
    '''
    def power_init(self):
        p = np.zeros(shape=(self.edge_n, self.user_n))
        for i in range(self.edge_n):
            p[i] = np.random.randn(self.user_n)
        return p
    def power_update(self, last_p, connect_flag):
        p = last_p * connect_flag/sum(last_p * connect_flag)*self.power_total
        return p

    def compute_DataRate(self, h, p):
        SINR = np.zeros(shape=(self.edge_n, self.user_n))
        DataRate = np.zeros(shape=(self.edge_n, self.user_n))
        max_sort_index = np.argsort(h)
        for i in range(self.edge_n):
            for j in range(self.user_n):
                sum = 0
                new_index = np.where(max_sort_index[i] == j)[0][0]
                for index in range(new_index + 1, self.user_n):
                    sum += (h[i][max_sort_index[i][index]] * p[i][max_sort_index[i][index]]) ** 2
                SINR[i][j] = (h[i][j] * p[i][j]) ** 2 / (sum + self.compute_noise(1))
                DataRate[i][j] = self.bandwidth * np.log2(1 + SINR[i][j])

        return np.sum(DataRate)

    def find_ST_edge(self, user_index, connect_f, h_, p_ini):
        edge_index = np.where(connect_f[:, user_index] == 1)[0]
        edge_rate = np.zeros(len(edge_index))
        connect_flag_ = copy.deepcopy(connect_f)
        for k in range(len(edge_index)):
            connect_f = copy.deepcopy(connect_flag_)
            connect_f[:, user_index] = 0
            connect_f[k][user_index] = 1
            power = self.power_update(p_ini, connect_f)
            edge_rate[k] = self.compute_DataRate(h_, power)
        max_edge_index = np.argmax(edge_rate)
        connect_flag_[:, user_index] = 0
        connect_flag_[max_edge_index][user_index] = 1
        max_rate = max(edge_rate)
        return max_rate, connect_flag_


DR = DataRate()
alpha = 4
user_num = 3
connect_flag = np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1]
])
'''random position'''
ex = np.zeros(DR.user_n)
ey = np.zeros(DR.user_n)
ex[:user_num], ey[:user_num] = DR.position_init(DR.xy[0], user_num)
ex[user_num:], ey[user_num:] = DR.position_init(DR.xy[1], user_num)
'''random channel gain'''
h = DR.channel_gain(alpha, ex, ey)
'''random init power'''
p_init = DR.power_init()

'''when update connect flag, we need to update power and rate'''
sj_rate = np.zeros(2)
js_rate = np.zeros(2)
ss_rate = np.zeros(4)

p = DR.power_update(p_init, connect_flag)
jj_rate = DR.compute_DataRate(h, p)
print("JT JT_rate:", jj_rate)

connect_flag = np.array([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1]
])
p = DR.power_update(p_init, connect_flag)
js_rate[0] = DR.compute_DataRate(h, p)
print("JT ST_rate:", js_rate[0])

connect_flag = np.array([
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1]
])
p = DR.power_update(p_init, connect_flag)
ss_rate[0] = DR.compute_DataRate(h, p)
print("ST ST_rate:", ss_rate[0])

connect_flag = np.array([
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 1]
])
p = DR.power_update(p_init, connect_flag)
sj_rate[0] = DR.compute_DataRate(h, p)
print("ST JT_rate:", sj_rate[0])

connect_flag = np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 1]
])
p = DR.power_update(p_init, connect_flag)
js_rate[1] = DR.compute_DataRate(h, p)
print("JT ST_rate:", js_rate[1])

connect_flag = np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1]
])
p = DR.power_update(p_init, connect_flag)
sj_rate[1] = DR.compute_DataRate(h, p)
print("ST JT_rate:", sj_rate[1])

connect_flag = np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1]
])
p = DR.power_update(p_init, connect_flag)
ss_rate[1] = DR.compute_DataRate(h, p)
print("ST ST_rate:", ss_rate[1])

connect_flag = np.array([
    [1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1]
])
p = DR.power_update(p_init, connect_flag)
ss_rate[2] = DR.compute_DataRate(h, p)
print("ST ST_rate:", ss_rate[2])

connect_flag = np.array([
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1]
])
p = DR.power_update(p_init, connect_flag)
ss_rate[3] = DR.compute_DataRate(h, p)
print("ST ST_rate:", ss_rate[3])

js_rate = max(js_rate)
sj_rate = max(sj_rate)
ss_rate = max(ss_rate)



connect_flag = np.array([
    [1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1]
])
cross_users_index = np.where(sum(connect_flag) > 1)[0]
trans_way = np.zeros(len(cross_users_index))
alpha_1 = [1] * len(cross_users_index)
beta_1 = [1] * len(cross_users_index)
alpha_2 = [1] * len(cross_users_index)
beta_2 = [1] * len(cross_users_index)

JT_connect_flag = copy.deepcopy(connect_flag)
o_connect_flag = copy.deepcopy(connect_flag)
rate = 0
for step_i in range(10):
    print("step", step_i, "---rate:", rate)
    for i in range(len(cross_users_index)):
        # sample
        st = np.random.beta(alpha_1[i], beta_1[i])
        jt = np.random.beta(alpha_2[i], beta_2[i])
        # choose arm
        if st > jt:
            # change trans_flag
            trans_way[i] = 0
            # 看连接哪一个edge server
            connect_flag[:, cross_users_index[i]] = JT_connect_flag[:, cross_users_index[i]]
            rate, connect_flag = DR.find_ST_edge(cross_users_index[i], connect_flag, h, p_init)
            # 相反选择 JT
            o_connect_flag[:, cross_users_index[i]] = JT_connect_flag[:, cross_users_index[i]]
            p = DR.power_update(p_init, o_connect_flag)
            o_rate = DR.compute_DataRate(h, p)
        else:
            trans_way[i] = 1
            connect_flag[:, cross_users_index[i]] = JT_connect_flag[:, cross_users_index[i]]
            p = DR.power_update(p_init, connect_flag)
            rate = DR.compute_DataRate(h, p)

            # 相反选择
            o_rate, _ = DR.find_ST_edge(cross_users_index[i], connect_flag, h, p_init)

        if rate > o_rate:
            if trans_way[i] == 0:
                alpha_1[i] += 1
            else:
                alpha_2[i] += 1
        else:
            if trans_way[i] == 0:
                beta_1[i] += 1
            else:
                beta_2[i] += 1


print("max_trans:", max(jj_rate,sj_rate,js_rate,ss_rate))
print(max(jj_rate,sj_rate,js_rate,ss_rate) == rate)