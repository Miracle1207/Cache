import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
    def __init__(self, user_n=20, edge_n=3):
        self.user_n = user_n
        self.edge_n = edge_n
        self.power_total = 19.953
        self.bandwidth = 4500000
        self.r = 1
        self.xy = np.array([[0, 0], [1, 0], [0.5, 1]])

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
            p[i] = abs(np.random.randn(self.user_n))
        return p
    def power_update(self, last_p, connect_flag):
        p = (last_p * connect_flag+0.0001)/(np.sum(last_p * connect_flag)+0.0001)*self.power_total
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

        return np.sum(DataRate, axis=1)

    def find_ST_edge(self, user_index, connect_f, h_, p_ini):
        edge_index = np.where(connect_f[:, user_index] == 1)[0]
        edge_rate = np.zeros(len(edge_index))
        connect_flag_ = copy.deepcopy(connect_f)
        for k in range(len(edge_index)):
            connect_f = copy.deepcopy(connect_flag_)
            connect_f[:, user_index] = 0
            connect_f[k][user_index] = 1
            power = self.power_update(last_p=p_ini, connect_flag=connect_f)
            edge_rate[k] = sum(self.compute_DataRate(h=h_, p=power))
        max_edge_index = np.argmax(edge_rate)
        connect_flag_[:, user_index] = 0
        connect_flag_[max_edge_index][user_index] = 1
        max_rate = max(edge_rate)
        return max_rate, connect_flag_
    def compute_correct_answer(self, c_connect_flag, h, p_ini):
        '''standard connect'''
        standard_connect_flag = copy.deepcopy(c_connect_flag)
        cross_users = np.where(sum(c_connect_flag) > 1)[0]
        '''给各种传输方式分类储存，最后求出最大的rate和对应的connect flag'''
        trans_rate = []
        trans_connect_set = []
        for i in range(2**(len(cross_users))):
            trans_rate.append([])
            trans_connect_set.append([])
        '''记录cross users选择的传输方式'''
        # trans_flag = [0 for i in range(len(cross_users))]
        # con_store = np.zeros(DR.edge_n * len(cross_users))
        rate_ = np.zeros(2 ** (self.edge_n * len(cross_users)))
        for i in range(2 ** (self.edge_n * len(cross_users))):
            c_connect_flag = copy.deepcopy(standard_connect_flag)
            trans_flag = [0 for i in range(len(cross_users))]
            con_store = np.zeros(self.edge_n * len(cross_users))
            str_i = bin(i).lstrip('0b')
            for j in range(len(str_i)):
                con_store[len(con_store) - 1 - j] = int(str_i[len(str_i) - 1 - j])
            for j in range(self.edge_n):
                for k in range(len(cross_users)):
                    c_connect_flag[j][cross_users[k]] = con_store[j * len(cross_users) + k]

            trans_index = 0
            for k in range(len(cross_users)):
                if np.sum(c_connect_flag, axis=0)[cross_users[k]] == 0:
                    rate_[i] = 0
                else:
                    if np.sum(c_connect_flag, axis=0)[cross_users[k]] > 1:
                        trans_flag[k] = 1
                    power = self.power_update(last_p=p_ini, connect_flag=c_connect_flag)
                    rate_[i] = sum(self.compute_DataRate(h=h, p=power))
                trans_index += int(trans_flag[k] * (2 ** k))
            trans_rate[trans_index].append(rate_[i])
            trans_connect_set[trans_index].append(c_connect_flag)

        correct_max_rate = max(rate_)
        for i in range(2**(len(cross_users))):
            trans_connect_set[i] = trans_connect_set[i][trans_rate[i].index(max(trans_rate[i]))]
            trans_rate[i] = max(trans_rate[i])
        return correct_max_rate, trans_rate, trans_connect_set
    '''input: 交叉用户的选择方式；
        output: 对应排列组合的最大datarate
        trans_list: 交叉用户选择的传输方式
        p_ini: initial power
        h_: channel gain
        c_connect_flag: cache flag 缓存后的标记
        cross_u_index: 交叉用户的index'''
    def find_max_rate(self, trans_list, cross_u_index, c_connect_flag, p_ini, h_):
        case_num = 1
        s_user = []
        for u_i in range(len(trans_list)):
            if trans_list[u_i] == 0:
                case_num *= int(np.sum(c_connect_flag, axis=0)[cross_u_index[u_i]])
                s_user.append(u_i)
        su_edge_index = np.zeros(len(s_user))
        rate_ = []
        flag = []
        for case_i in range(case_num):
            temp = case_i
            for su_i in range(len(s_user)):
                su_edge_index[su_i] = int(temp % int(np.sum(c_connect_flag, axis=0)[cross_u_index[su_i]]))
                temp = temp / int(np.sum(c_connect_flag, axis=0)[cross_u_index[su_i]])
                c_connect_flag[:, cross_u_index[su_i]] = 0
                c_connect_flag[int(su_edge_index[su_i])][cross_u_index[su_i]] = 1
            power = self.power_update(last_p=p_ini, connect_flag=c_connect_flag)
            rate_.append(sum(self.compute_DataRate(h=h_, p=power)))
            flag.append(copy.deepcopy(c_connect_flag))
        max_flag = flag[rate_.index(max(rate_))]
        max_rate = max(rate_)
        return max_rate, max_flag


    def run_BLA(self, bla_steps, connect_flag, p_init, h):
        '''
        BLA: to choose optimal access way
        bla_steps: 需要多少步收敛
        connect_flag: 来自cache网络的缓存命中标记
        '''
        cross_users_index = np.where(sum(connect_flag) > 1)[0]
        trans_way = np.zeros(len(cross_users_index))
        alpha_1 = [1] * len(cross_users_index)
        beta_1 = [1] * len(cross_users_index)
        alpha_2 = [1] * len(cross_users_index)
        beta_2 = [1] * len(cross_users_index)

        # correct_max_rate, trans_rate, trans_cache_flag = self.compute_correct_answer(c_connect_flag=connect_flag, h=h, p_ini=p_init)

        JT_connect_flag = copy.deepcopy(connect_flag)
        o_connect_flag = copy.deepcopy(connect_flag)
        rate = 0

        for step_i in range(bla_steps):
            for i in range(len(cross_users_index)):
                # sample
                st = np.random.beta(alpha_1[i], beta_1[i])
                jt = np.random.beta(alpha_2[i], beta_2[i])
                # choose arm
                if st > jt:
                    '''find ST edge--little error'''
                    # # change trans_flag
                    # trans_way[i] = 0
                    # # 看连接哪一个edge server
                    # # 需要在ST组合中找到最小的ST ST
                    # connect_flag[:, cross_users_index[i]] = JT_connect_flag[:, cross_users_index[i]]
                    # rate, connect_flag = self.find_ST_edge(user_index=cross_users_index[i],
                    #                                        connect_f=connect_flag, h_=h, p_ini=p_init)
                    #
                    # # 相反选择 JT
                    # o_connect_flag[:, cross_users_index[i]] = JT_connect_flag[:, cross_users_index[i]]
                    # p = self.power_update(last_p=p_init, connect_flag=o_connect_flag)
                    # o_rate = sum(self.compute_DataRate(h=h, p=p))
                    '''compute all cases--too slow'''
                    # # 相反选择
                    # trans_way[i] = 1
                    # trans_index = 0
                    # for j in range(len(cross_users_index)):
                    #     trans_index += int(trans_way[j] * (2 ** j))
                    # o_rate = trans_rate[trans_index]
                    # # 本次选择
                    # trans_way[i] = 0
                    # trans_index = 0
                    # for j in range(len(cross_users_index)):
                    #     trans_index += int(trans_way[j] * (2 ** j))
                    # rate = trans_rate[trans_index]
                    # connect_flag = trans_cache_flag[trans_index]
                    '''compute partial cases'''
                    # 相反选择
                    trans_way[i] = 1
                    o_rate, _ = self.find_max_rate(trans_list=trans_way,
                                                            cross_u_index=cross_users_index,
                                                            c_connect_flag=connect_flag, p_ini=p_init, h_=h)
                    # 本次选择
                    trans_way[i] = 0
                    rate, connect_flag = self.find_max_rate(trans_list=trans_way,
                                                            cross_u_index=cross_users_index,
                                                            c_connect_flag=connect_flag, p_ini=p_init, h_=h)

                else:
                    '''find ST edge--little error'''
                    # trans_way[i] = 1
                    # connect_flag[:, cross_users_index[i]] = JT_connect_flag[:, cross_users_index[i]]
                    # p = self.power_update(last_p=p_init, connect_flag=o_connect_flag)
                    # rate = sum(self.compute_DataRate(h=h, p=p))
                    #
                    # # 相反选择
                    # o_rate, _ = self.find_ST_edge(user_index=cross_users_index[i],
                    #                                        connect_f=connect_flag, h_=h, p_ini=p_init)
                    '''compute all cases--too slow'''
                    # # 相反选择
                    # trans_way[i] = 0
                    # trans_index = 0
                    # for j in range(len(cross_users_index)):
                    #     trans_index += int(trans_way[j] * (2 ** j))
                    # o_rate = trans_rate[trans_index]
                    # # 本次选择
                    # trans_way[i] = 1
                    # trans_index = 0
                    # for j in range(len(cross_users_index)):
                    #     trans_index += int(trans_way[j] * (2 ** j))
                    # rate = trans_rate[trans_index]
                    # connect_flag = trans_cache_flag[trans_index]
                    '''compute partial cases'''
                    # 相反选择
                    trans_way[i] = 0
                    o_rate, _ = self.find_max_rate(trans_list=trans_way,
                                                   cross_u_index=cross_users_index,
                                                   c_connect_flag=connect_flag, p_ini=p_init, h_=h)
                    # 本次选择
                    trans_way[i] = 1
                    rate, connect_flag = self.find_max_rate(trans_list=trans_way,
                                                            cross_u_index=cross_users_index,
                                                            c_connect_flag=connect_flag, p_ini=p_init, h_=h)

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
            print("step", step_i, "---rate:", rate)

        return connect_flag




