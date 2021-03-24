import gym
import numpy as np
import random
import math
import copy
from decimal import Decimal
from gym import error, spaces, utils
from gym.utils import seeding
from gym_cache.envs.cache_method import CacheMethod
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class CacheEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.task_n = 10  # the number of total tasks
        self.task = list(range(self.task_n))  # the total tasks set

        self.user_n = 10  # the number of users
        self.each_user_task_n = 1  # the number of requested task by each user
        self.users_tasks = np.zeros(shape=(self.user_n, self.each_user_task_n))
        for i in range(self.user_n):  # initial users_tasks randomly
            self.users_tasks[i] = CacheMethod.Zipf_sample(task_set=self.task, num=self.each_user_task_n)
        self.edge_n = 3
        # self.edge_server = 3
        # self.edge_n = 7  # the number of agent(edge server)
        self.each_edge_cache_n = 5  # the number of caching tasks by each edge server
        # state : the set of each edge servers' caching tasks
        self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))

        # state = serve_success+flag   agent         user_task     自己缓存的内容的大小
        self.state = np.zeros(
            shape=(self.edge_n, (self.each_edge_cache_n + self.each_edge_cache_n * self.edge_n + self.user_n)))
        # cache_flag = np.array(
        #     [[0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
        #      [1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        #      [1, 1, 1, 0, 1, 0, 0, 1, 1, 1]]).reshape(1, 30)
        # self.cache_flag = np.repeat(cache_flag, 7, axis=0)
        # self.state = self.cache_flag
        self.seed()  # TODO control some features change randomly
        self.viewer = None  # whether open the visual tool
        self.steps_beyond_done = None  # current step = 0
        self.step_num = 0
        # bla
        self.trans_t = 5
        self.cache_t = 10
        self.buffer_t = 1
        self.alpha_1 = []
        self.beta_1 = []
        self.alpha_2 = []
        self.beta_2 = []
        self.cache_buffer_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.change_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.last_25_cache_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.last_25_change_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        # PPP
        self.lambda0 = 2
        #self.r = np.sqrt(1 / np.pi)
        self.r = 1
        # power bisection
        self.p_total = 19.953  # total power is 19.953w, namely 43dbm
        self.p = np.random.uniform(0,1, self.edge_n * self.user_n)
        self.p = np.reshape(self.p, (self.edge_n, self.user_n))
        # channel bandwidth
        self.bandwidth = 4500000  # 4.5MHZ
        self.h = np.zeros(shape=(self.edge_n, self.user_n))
        for i in range(self.edge_n):
            self.h[i] = abs(1 / np.sqrt(2) * (np.random.randn(self.user_n) + 1j * np.random.randn(self.user_n)))

    @property
    def observation_space(self):
        return [spaces.Box(low=0, high=self.task_n,
                           shape=(self.each_edge_cache_n + self.each_edge_cache_n * self.edge_n + self.user_n, 1)) for i
                in range(self.edge_n)]
        # return [spaces.Box(low=0, high=self.task_n, shape=(3 * 10 , 1)) for i
        #         in range(self.edge_n)]

    @property
    def action_space(self):  # 选择1个位置，从10个任务中选一个进行缓存更换 + 一种情况不缓存
        return [spaces.Discrete(31) for i in range(self.edge_n)]

    @property
    def agents(self):
        return ['agent' for i in range(self.edge_n)]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # sample randomly from task_set
    def random_sample(self, task_set, num):
        sample_task = random.sample(task_set, num)
        return sample_task

    def action_wrapper(self, acs):
        acs = acs[0]
        ac_num = np.zeros(len(acs))
        ac_task = np.zeros(len(acs))
        for i in range(len(acs)):
            action = list(acs[i]).index(1)
            # 不更改
            if action == 30:
                return 0
            # 更改缓存
            else:
                ac_num[i] = action // 10
                ac_task[i] = action % 10
        return ac_num, ac_task

    def max_cache_num(self, user_task):
        uni_len = len(np.unique(user_task))
        if uni_len == 5:
            max_score = 3
        elif uni_len == 4:
            max_score = 4
        else:
            max_score = 5
        return max_score

    # Gaussian noise
    def compute_noise(self, NUM_Channel):
        ThermalNoisedBm = -174  # -174dBm/Hz
        var_noise = 10 ** ((ThermalNoisedBm - 30) / 10) * self.bandwidth / (
            NUM_Channel)  # envoriment noise is 1.9905e-15
        return var_noise

    '''
  compute signal to inference plus noise ratio
  h: channel gain from edge server e to all users with the shape of (1, self.user_n)
  x: serve success from edge server e to all users with the shape of (1, self.user_n)
  sinr: SINR from e to all users
  '''
    '''
    for one user, compute the sinr of it
    users_n: 1
    h: the channel gain that from each edge serve to this user h_n = self.edge_n
    p: the power that from each edge server to this user p_n = self.edge_n
    edge_index: the index of edge server that serves this user
    if len(edge_index)=1, mode="ST"-single point transmission;
    if len(edge_index)=2 or 3, mode="JT"-joint transmission
    '''
    def compute_SINR(self, h, p, serve_index):
        max_sort_index = np.argsort(p)
        sinr_sum = 0
        for i in range(self.edge_n):
            sinr_sum += h[i] * p[i]
        # ST
        if len(serve_index) == 1:
            sinr = h[serve_index]*p[serve_index]/(sinr_sum - h[serve_index]*p[serve_index])
            return sinr
        elif len(serve_index) == 2:
            sinr1 = h[max_sort_index[0]]*p[max_sort_index[0]]/(sinr_sum - h[max_sort_index[0]]*p[max_sort_index[0]])
            sinr2 = h[max_sort_index[1]]*p[max_sort_index[1]]/(sinr_sum - h[max_sort_index[0]]*p[max_sort_index[0]]-h[max_sort_index[1]]*p[max_sort_index[1]])
            return sinr1+sinr2
        elif len(serve_index) == 3:
            sinr1 = h[max_sort_index[0]]*p[max_sort_index[0]]/(sinr_sum - h[max_sort_index[0]]*p[max_sort_index[0]])
            sinr2 = h[max_sort_index[1]]*p[max_sort_index[1]]/(sinr_sum - h[max_sort_index[0]]*p[max_sort_index[0]]
                                                               -h[max_sort_index[1]]*p[max_sort_index[1]])
            sinr3 = h[max_sort_index[2]]*p[max_sort_index[2]]/(sinr_sum - h[max_sort_index[0]]*p[max_sort_index[0]]
                                                               -h[max_sort_index[1]]*p[max_sort_index[1]]
                                                               -h[max_sort_index[2]]*p[max_sort_index[2]])

            return sinr1+sinr2+sinr3
    '''
    h,p,trans_flag: all users served by one edge server
    
    '''
    def compute_DataRate(self, trans_flag):
        h = self.h
        p = self.p * trans_flag / np.sum(self.p * trans_flag) * self.p_total
        SINR = np.zeros(shape=(self.edge_n, self.user_n))
        DataRate = np.zeros(shape=(self.edge_n, self.user_n))
        max_sort_index = np.argsort(p)
        for i in range(self.edge_server):
            for j in range(self.user_n):
                sum = 0
                new_index = np.where(max_sort_index == j)[0][0]
                for index in range(new_index + 1, self.user_n):
                    sum += (h[i][max_sort_index[i][index]] * p[i][max_sort_index[i][index]]) ** 2
                SINR[i][j] = (h[i][j] * p[i][j]) ** 2 / (sum + self.compute_noise(1))
                DataRate[i][j] = self.bandwidth * np.log2(1 + SINR[i][j])

        return np.sum(DataRate)

    #def joint_transmission(self, ):

    '''
  compute downlink rate
  sinr: SINR from an edge server to all users with the shape of (1, self.user_n)
  '''

    def compute_Rate(self, sinr):
        rate= self.bandwidth * np.log2(1 + sinr)
        return rate

    '''
    add BLA:
    beta_compute: generate probability of action
    reward_compute: compute the delay of action
    '''
    # # 从beta分布中生成俩值
    # def beta_compute(self, a_1, b_1, a_2, b_2):
    #     temp = 0
    #     for i in range(int(a_2), int(a_2 + b_2)):
    #         x_1 = Decimal(math.factorial(i))
    #         x_2 = Decimal(math.factorial(a_2 + b_2 - 1 - i))
    #         x_3 = Decimal(math.factorial(a_1 + i - 1))
    #         x_4 = Decimal(math.factorial(a_2 + b_2 + b_1 - i - 2))
    #         temp += (x_3 * x_4) / (x_1 * x_2)
    #     x_5 = Decimal(math.factorial(a_1 + b_1 - 1))
    #     x_6 = Decimal(math.factorial(a_2 + b_2 - 1))
    #     x_7 = Decimal(math.factorial(a_1 - 1))
    #     x_8 = Decimal(math.factorial(b_1 - 1))
    #     x_9 = Decimal(math.factorial(a_1 + b_1 + a_2 + b_2 - 2))
    #     p = ((x_5 * x_6) / ( x_7 * x_8 * x_9)) * temp
    #     return p

    # def reward_compute(self, action, change_flag):
    #     delay = self.trans_t # t = 5
    #     if action == 0: # cache
    #         if change_flag == 1:
    #             delay += self.cache_t # t = 10
    #     else: # buffer
    #         if change_flag == 1: # t = 1
    #             delay += self.buffer_t
    #     return delay
    '''
    PPP: poisson point process
    user_ppp: input the coordinate(xx0, yy0) of edge server, 
                generate 5 user around it and draw the area of working
                lambda0: the intensity (ie mean density) of the Poisson process
    '''
    def user_ppp(self, xy0, lambda0):
        # Point process parameters
        # lambda0 users in each area of np.pi* self.r **2
        numbPoints = np.random.poisson(lambda0)  # Poisson number of points
        theta = 2 * np.pi * np.random.uniform(0, 1, numbPoints)  # angular coordinates
        rho = self.r * np.sqrt(np.random.uniform(0, 1, numbPoints))  # radial coordinates
        # Convert from polar to Cartesian coordinates
        xx = rho * np.cos(theta)
        yy = rho * np.sin(theta)
        # Shift centre of disk to (xx0,yy0)
        xx = xx + xy0[0]
        yy = yy + xy0[1]
        return xx, yy, numbPoints
    '''
    generate the coordinate of users and mark users in each area of edge server
    '''
    def user_task(self):
        xy = np.array([[0, 0], [1, 0], [1, 1.5]])
        user_n = np.zeros(self.edge_n)
        u_x1, u_y1, user_n[0] = self.user_ppp(xy[0], self.lambda0)
        u_x2, u_y2, user_n[1] = self.user_ppp(xy[1], self.lambda0)
        u_x3, u_y3, user_n[2] = self.user_ppp(xy[2], self.lambda0)
        u_x = np.r_[u_x1, u_x2, u_x3]
        u_y = np.r_[u_y1, u_y2, u_y3]
        all_user_n = int(sum(user_n))
        user_in_edge = np.zeros(shape=(self.edge_n, all_user_n))
        for i in range(all_user_n):
            if pow((u_x[i]-0), 2) + pow((u_y[i]-0), 2) <= pow(self.r, 2):
                user_in_edge[0][i] = 1
            if pow((u_x[i]-1), 2) + pow((u_y[i]-0), 2) <= pow(self.r, 2):
                user_in_edge[1][i] = 1
            if pow((u_x[i]-1), 2) + pow((u_y[i]-1.5), 2) <= pow(self.r, 2):
                user_in_edge[2][i] = 1
        fig = plt.figure()
        # Plotting
        plt.scatter(u_x1, u_y1, edgecolor='b', facecolor='none', alpha=0.5)
        plt.scatter(u_x2, u_y2, edgecolor='g', facecolor='none', alpha=0.5)
        plt.scatter(u_x3, u_y3, edgecolor='r', facecolor='none', alpha=0.5)
        # margin

        ax = fig.add_subplot(111)
        cir1 = Circle(xy=(0.0, 0.0), radius=self.r, alpha=0.1, color="b")
        cir2 = Circle(xy=(1.0, 0.0), radius=self.r, alpha=0.1, color="g")
        cir3 = Circle(xy=(1.0, 1.5), radius=self.r, alpha=0.1, color="r")
        ax.add_patch(cir1)
        ax.add_patch(cir2)
        ax.add_patch(cir3)

        ax.plot(0, 0, 'b*')
        ax.plot(1, 0, 'g*')
        ax.plot(1, 1.5, 'r*')

        plt.axis('scaled')
        plt.axis('equal')  # changes limits of x or y axis so that equal increments of x and y have the same length
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.show()
        return user_in_edge, all_user_n

    def action_10_2(self, index, ac_len):
      ac = np.zeros(ac_len)
      ac_str = bin(index).lstrip('0b')
      for j in range(len(ac_str)):
          ac[ac_len - 1 - j] = int(ac_str[len(ac_str) - 1 - j])
      return ac

    def step(self, action):
        # RL: cache network
        self.step_num += 1
        for i in range(self.user_n):  # initial users_tasks randomly
            self.users_tasks[i] = CacheMethod.Zipf_sample(task_set=self.task, num=self.each_user_task_n)
        cache_flag = np.zeros(shape=(self.edge_n, self.user_n))
        if self.action_wrapper(action) != 0:
            ac_num, ac_task = self.action_wrapper(action)
            for i in range(self.edge_n):
                self.edge_caching_task[i][int(ac_num[i])] = ac_task[i]

        for i in range(self.edge_n):
            for j in range(self.user_n):
                if self.users_tasks[j] in self.edge_caching_task[i]:
                    cache_flag[i][j] = 1

        fail = list(sum(cache_flag)).count(0)
        # cache hit rate
        reward = np.zeros(self.edge_n)
        for k in range(self.edge_n):
            reward[k] = 1/3*(1-fail/self.user_n)
        # change state
        for i in range(self.edge_n):
            self.state[i] = np.append(np.append(self.edge_caching_task[i], np.ndarray.flatten(self.edge_caching_task)),
                                      self.users_tasks)
        if self.step_num == 24:
            print("edge:\n", cache_flag)

        done = 1
        next_obs = self.state
        return next_obs, reward, done, {}

    def reset(self):
        self.step_num = 0
        # cache_user_fail = np.zeros(self.user_n)
        for i in range(self.user_n):  # initial users_tasks randomly
            self.users_tasks[i] = CacheMethod.Zipf_sample(self.task, self.each_user_task_n)
        # update the state at the beginning of each episode
        self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.state = np.zeros(
            shape=(self.edge_n, (self.each_edge_cache_n + self.each_edge_cache_n * self.edge_n + self.user_n)))
        # self.state = self.cache_flag
        self.steps_beyond_done = None  # set the current step as 0
        return np.array(self.state)  # return the initial state

    def render(self, mode='human'):
        ...

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


ACTION = {
    0: '0000',
    1: '0001',
    2: '0010',
    3: '0011',
    4: '0100',
    5: '0101',
    6: '0110',
    7: '0111',
    8: '1000',
    9: '1001',
    10:'1010',
    11:'1011',
    12:'1100',
    13:'1101',
    14:'1110',
    15:'1111'
}
