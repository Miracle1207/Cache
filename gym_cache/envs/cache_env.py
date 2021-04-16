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
from gym_cache.envs.DataRate import DataRate


class CacheEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed_num=2):
        '''set random seed'''
        np.random.seed(seed_num)
        random.seed(seed_num)

        self.DR = DataRate(seed_num=seed_num)
        self.task_n = 50  # the number of total tasks
        self.task = list(range(self.task_n))  # the total tasks set
        self.CM = CacheMethod()
        self.user_n = 20  # the number of users
        self.each_user_task_n = 1  # the number of requested task by each user
        self.users_tasks = np.zeros(shape=(self.user_n, self.each_user_task_n))
        # for i in range(self.user_n):  # initial users_tasks randomly
        self.users_tasks = CacheMethod.Zipf_sample(self.CM, task_set=self.task, num=self.user_n*self.each_user_task_n, v=1.2) # v = 0.8
        self.user_recent_q = np.zeros(self.task_n)
        self.edge_n = 3
        self.each_edge_cache_n = 10  # the number of caching tasks by each edge server
        # state : the set of each edge servers' caching tasks
        self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.file_size = 1000000  # 1 Mbit

        '''users' position'''
        self.user_x = np.loadtxt("/home/mqr/Cache/gym_cache/envs/user_ppp/1221/ex.txt")
        self.user_y = np.loadtxt("/home/mqr/Cache/gym_cache/envs/user_ppp/1221/ey.txt")
        '''random channel gain'''
        self.alpha = 4
        self.edge_h = self.DR.channel_gain(self.alpha, self.user_x, self.user_y)
        self.cloud_h = self.DR.cloud_channel_gain(self.alpha)
        '''random init power'''
        self.edge_p_init = self.DR.power_init()
        self.cloud_p_init = self.DR.cloud_power_init()

        '''distance from edge server i to user j'''
        self.d = DataRate.distance_compute(self.DR, self.user_x, self.user_y)
        self.position_flag = np.zeros(shape=(self.edge_n, self.user_n))
        '''position'''
        for i in range(self.edge_n):
            for j in range(self.user_n):
                if self.d[i][j] < self.DR.r:
                    self.position_flag[i][j] = 1
        self.each_edge_user_num = np.sum(self.position_flag, axis=1)
        self.obsp_user_task = int(max(self.each_edge_user_num))
        self.each_edge_user_task = np.zeros(shape=(self.edge_n, self.obsp_user_task))
        '''state: own users' request + caching contents'''
        self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        # self.state = np.zeros(
        #     shape=(self.edge_n, self.obsp_user_task))
        self.state = np.zeros(
            shape=(self.edge_n, self.each_edge_cache_n+self.obsp_user_task))
        '''state update'''
        for i in range(self.edge_n):
            for j in range(self.obsp_user_task):
                if j < self.each_edge_user_num[i]:
                    self.each_edge_user_task[i][j] = self.users_tasks[np.where(self.position_flag[i] == 1)[0][j]]
                else:
                    self.each_edge_user_task[i][j] = self.each_edge_user_task[i][j-int(self.each_edge_user_num[i])]
        '''自缓存+自用户'''
        for i in range(self.edge_n):
            self.state[i] = np.append(self.edge_caching_task[i], self.each_edge_user_task[i])
        '''自用户'''
        # for i in range(self.edge_n):
        #     self.state[i] = self.each_edge_user_task[i]
        '''全用户'''
        # for i in range(self.edge_n):
        #     self.state[i] = self.users_tasks.T
         # TODO control some features change randomly
        self.viewer = None  # whether open the visual tool
        self.steps_beyond_done = None  # current step = 0
        self.step_num = 0

        '''plot'''
        # fig = plt.figure()
        # # Plotting
        # plt.scatter(self.user_x, self.user_y, edgecolor='b', facecolor='none', alpha=0.5, label="Users")
        #
        # # margin
        #
        # ax = fig.add_subplot(111)
        # cir1 = Circle(xy=(0.0, 0.0), radius=1, alpha=0.1, color="b", label="E1 Coverage")
        # cir2 = Circle(xy=(1.0, 0.0), radius=1, alpha=0.1, color="g", label="E2 Coverage")
        # cir3 = Circle(xy=(0.5, 1), radius=1, alpha=0.1, color="r", label="E3 Coverage")
        # ax.add_patch(cir1)
        # ax.add_patch(cir2)
        # ax.add_patch(cir3)
        #
        # ax.plot(0, 0, 'b*', label="Edge 1")
        # ax.plot(1, 0, 'g*', label="Edge 2")
        # ax.plot(0.5, 1, 'r*', label="Edge 3")
        #
        # plt.axis('scaled')
        # plt.axis('equal')  # changes limits of x or y axis so that equal increments of x and y have the same length
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.axis('equal')
        # plt.legend(loc="upper right",borderaxespad=0)
        # plt.show()


    @property
    def observation_space(self):
        return [spaces.Box(low=0, high=self.task_n,
                           shape=(self.obsp_user_task+self.each_edge_cache_n, 1)) for i
                in range(self.edge_n)]

    @property
    def action_space(self):  # agent选一个进来
        return [spaces.Discrete(self.each_edge_cache_n*self.task_n+1) for i in range(self.edge_n)]

    @property
    def agents(self):
        return ['agent' for i in range(self.edge_n)]

    # sample randomly from task_set
    def random_sample(self, task_set, num):
        sample_task = random.sample(task_set, num)
        return sample_task

    def action_wrapper(self, acs):
        acs = acs[0]
        ac_num = np.zeros(len(acs))
        ac_task = np.zeros(len(acs))
        for i in range(len(acs)):
            if sum(list(acs[i])) == 0:
                return 0
            else:
                action = list(acs[i]).index(1)
                # 不更改
                if action == self.each_edge_cache_n*self.task_n:
                    return 0
                # 更改缓存
                else:
                    ac_num[i] = action // self.task_n
                    ac_task[i] = action % self.task_n
                return ac_num, ac_task
    # def action_wrapper(self, acs):
    #     acs = acs[0]
    #     ac_task = np.zeros(len(acs))
    #     for i in range(len(acs)):
    #         action = list(acs[i]).index(1)
    #         ac_task[i] = action
    #     return ac_task


    def step(self, action):
        # RL: cache network
        self.step_num += 1
        self.users_tasks = CacheMethod.Zipf_sample(self.CM, task_set=self.task, num=self.user_n*self.each_user_task_n, v=1.2)
        cache_flag = np.zeros(shape=(self.edge_n, self.user_n))
        if self.step_num % 3 == 0:
            self.user_recent_q = np.zeros(self.task_n)
        for each in self.users_tasks:
            self.user_recent_q[each] += 1
        '''
        cache predict
        '''
        if self.action_wrapper(action) != 0:
            ac_num, ac_task = self.action_wrapper(action)
            for i in range(self.edge_n):
                self.edge_caching_task[i][int(ac_num[i])] = ac_task[i]
        '''
        cache predict + FIFO
        '''
        # ac_task = self.action_wrapper(action)
        # for i in range(self.edge_n):
        #     if ac_task[i] < self.task_n:
        #         self.edge_caching_task[i][self.each_edge_cache_n-1] = ac_task[i]
        '''
        cache predict + LRU
        '''
        # ac_task = self.action_wrapper(action)
        # for i in range(self.edge_n):
        #     if ac_task[i] < self.task_n:
        #         self.edge_caching_task[i] = CacheMethod.RL_LRU(self.CM, self.user_recent_q, self.edge_caching_task[i], ac_task[i])
        '''
        FIFO
        '''
        # temp = self.edge_caching_task[:, 1:]
        # new_task = np.random.choice(self.task, self.edge_n)
        # self.edge_caching_task = np.column_stack((temp, new_task.T))

        '''random+LRU'''
        # self.edge_caching_task = CacheMethod.random_LRU(self.CM, self.user_recent_q, self.edge_caching_task, self.task)
        '''random+LFU'''
        # self.edge_caching_task = CacheMethod.random_LFU(self.CM, self.edge_caching_task, self.task)
        '''
        cache hit rate
        '''
        # reward = np.zeros(self.edge_n)
        # cache_hit = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        # for i in range(self.edge_n):
        #     for j in range(self.each_edge_cache_n):
        #         if self.edge_caching_task[i][j] in self.users_tasks:
        #             cache_hit[i][j] = 1
        #     reward[i] = 1/3*sum(cache_hit[i])/self.each_edge_cache_n
        #

        '''
        user serve rate
        '''
        for i in range(self.edge_n):
            for j in range(self.user_n):
                if self.position_flag[i][j] == 1:
                    if self.users_tasks[j] in self.edge_caching_task[i]:
                        cache_flag[i][j] = 1

        # fail = list(sum(cache_flag)).count(0)
        # # cache hit rate
        # reward = np.zeros(self.edge_n)
        # for k in range(self.edge_n):
        #     reward[k] = 1/3*(1-fail/self.user_n)

        '''Data rate as reward'''
        # p = self.DR.power_update(self.edge_p_init, cache_flag)
        # rate = self.DR.compute_DataRate(self.edge_h, p)
        # # obs_user_n = self.each_edge_user_num
        # '''将reward设为 dara rate 比上 观察用户数量'''
        # # reward = rate/obs_user_n
        # reward = rate
        '''delay as reward'''

        delay, rate = self.DR.compute_Delay(cloud_h=self.cloud_h, cloud_p_ini=self.cloud_p_init,
                                      edge_h=self.edge_h, edge_p_ini=self.edge_p_init,
                                      connect_flag=cache_flag, file_size=self.file_size)
        reward = rate

        '''state update'''
        for i in range(self.edge_n):
            for j in range(self.obsp_user_task):
                if j < self.each_edge_user_num[i]:
                    self.each_edge_user_task[i][j] = self.users_tasks[np.where(self.position_flag[i] == 1)[0][j]]
                else:
                    self.each_edge_user_task[i][j] = self.each_edge_user_task[i][j-int(self.each_edge_user_num[i])]
        '''自缓存+自用户'''
        for i in range(self.edge_n):
            self.state[i] = np.append(self.edge_caching_task[i], self.each_edge_user_task[i])
        '''自用户'''
        # for i in range(self.edge_n):
        #     self.state[i] = self.each_edge_user_task[i]
        '''全用户'''
        # for i in range(self.edge_n):
        #     self.state[i] = self.users_tasks.T
        if self.step_num == 99:
            print("edge:\n", cache_flag)

        done = 1
        next_obs = self.state
        return next_obs, reward, done, {}, cache_flag, delay
    def bi_step(self, action):
        # RL: cache network
        self.step_num += 1
        cache_flag = np.zeros(shape=(self.edge_n, self.user_n))
        '''
        cache predict
        '''
        if self.action_wrapper(action) != 0:
            ac_num, ac_task = self.action_wrapper(action)
            for i in range(self.edge_n):
                self.edge_caching_task[i][int(ac_num[i])] = ac_task[i]

        '''
        user serve rate
        '''
        for i in range(self.edge_n):
            for j in range(self.user_n):
                if self.position_flag[i][j] == 1:
                    if self.users_tasks[j] in self.edge_caching_task[i]:
                        cache_flag[i][j] = 1

        '''Data rate as reward'''
        # p = self.DR.power_update(self.p_init, cache_flag)
        # rate = self.DR.compute_DataRate(self.edge_h, p)
        # obs_user_n = self.each_edge_user_num
        # '''将reward设为 dara rate 比上 观察用户数量'''
        # # reward = rate/obs_user_n
        # reward = rate
        '''Delay as reward'''
        delay, rate = self.DR.compute_Delay(cloud_h=self.cloud_h, cloud_p_ini=self.cloud_p_init,
                                      edge_h=self.edge_h, edge_p_ini=self.edge_p_init,
                                      connect_flag=cache_flag, file_size=self.file_size)
        reward = rate

        '''state update'''
        for i in range(self.edge_n):
            for j in range(self.obsp_user_task):
                if j < self.each_edge_user_num[i]:
                    self.each_edge_user_task[i][j] = self.users_tasks[np.where(self.position_flag[i] == 1)[0][j]]
                else:
                    self.each_edge_user_task[i][j] = self.each_edge_user_task[i][j-int(self.each_edge_user_num[i])]
        '''自缓存+自用户'''
        for i in range(self.edge_n):
            self.state[i] = np.append(self.edge_caching_task[i], self.each_edge_user_task[i])
        '''自用户'''
        # for i in range(self.edge_n):
        #     self.state[i] = self.each_edge_user_task[i]
        '''全用户'''
        # for i in range(self.edge_n):
        #     self.state[i] = self.users_tasks.T
        if self.step_num == 99:
            print("edge:\n", cache_flag)

        done = 1
        next_obs = self.state
        return next_obs, reward, done, {}, cache_flag, delay

    def reset(self):
        self.step_num = 0
        # cache_user_fail = np.zeros(self.user_n)
        # self.users_tasks = CacheMethod.Zipf_sample(self.CM, task_set=self.task, num=self.user_n*self.each_user_task_n, v=2)
        # update the state at the beginning of each episode
        self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.each_edge_user_task = np.zeros(shape=(self.edge_n, self.obsp_user_task))
        '''state: own users' request + caching contents'''
        self.state = np.zeros(
            shape=(self.edge_n, self.each_edge_cache_n+self.obsp_user_task))
        self.steps_beyond_done = None  # set the current step as 0
        return np.array(self.state)  # return the initial state

    def bi_reset(self, access_way):
        '''只改变state，变成接入用户的申请'''
        self.step_num = 0
        obs_user = [[] for u_i in range(self.edge_n)]
        for e_i in range(self.edge_n):
            for u_i in np.where(self.position_flag[e_i] == 1)[0]:
                if sum(access_way)[u_i] < 2:
                    obs_user[e_i].append(u_i)
                else:
                    if access_way[e_i][u_i] == 1:
                        obs_user[e_i].append(u_i)
            self.each_edge_user_num[e_i] = len(obs_user[e_i])
        self.each_edge_user_task = np.zeros(shape=(self.edge_n, self.obsp_user_task))
        '''state: own users' request + caching contents'''
        '''state update'''
        for i in range(self.edge_n):
            for j in range(self.obsp_user_task):
                if j < self.each_edge_user_num[i]:
                    self.each_edge_user_task[i][j] = self.users_tasks[obs_user[i][j]]
                else:
                    self.each_edge_user_task[i][j] = self.each_edge_user_task[i][j - int(self.each_edge_user_num[i])]
        '''state: own users' request + caching contents'''
        for i in range(self.edge_n):
            self.state[i] = np.append(self.edge_caching_task[i], self.each_edge_user_task[i])
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
