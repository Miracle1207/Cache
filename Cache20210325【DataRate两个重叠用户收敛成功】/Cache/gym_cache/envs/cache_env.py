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
        self.task_n = 100  # the number of total tasks
        self.task = list(range(self.task_n))  # the total tasks set

        self.user_n = 30  # the number of users
        self.each_user_task_n = 1  # the number of requested task by each user
        self.users_tasks = np.zeros(shape=(self.user_n, self.each_user_task_n))
        for i in range(self.user_n):  # initial users_tasks randomly
            self.users_tasks[i] = CacheMethod.Zipf_sample(task_set=self.task, num=self.each_user_task_n)

        self.edge_n = 3
        self.each_edge_cache_n = 10  # the number of caching tasks by each edge server
        # state : the set of each edge servers' caching tasks
        self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))

        # state = serve_success+flag   agent         user_task     自己缓存的内容的大小
        self.state = np.zeros(
            shape=(self.edge_n, (self.each_edge_cache_n + self.each_edge_cache_n * self.edge_n + self.user_n)))

        self.seed()  # TODO control some features change randomly
        self.viewer = None  # whether open the visual tool
        self.steps_beyond_done = None  # current step = 0
        self.step_num = 0


    @property
    def observation_space(self):
        return [spaces.Box(low=0, high=self.task_n,
                           shape=(self.each_edge_cache_n + self.each_edge_cache_n * self.edge_n + self.user_n, 1)) for i
                in range(self.edge_n)]
        # return [spaces.Box(low=0, high=self.task_n, shape=(3 * 10 , 1)) for i
        #         in range(self.edge_n)]

    @property
    def action_space(self):  # 选择1个位置，从10个任务中选一个进行缓存更换 + 一种情况不缓存
        return [spaces.Discrete(self.each_edge_cache_n*self.task_n+1) for i in range(self.edge_n)]

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
            if action == self.each_edge_cache_n*self.task_n:
                return 0
            # 更改缓存
            else:
                ac_num[i] = action // self.task_n
                ac_task[i] = action % self.task_n
        return ac_num, ac_task


    def step(self, action):
        # RL: cache network
        self.step_num += 1
        for i in range(self.user_n):  # initial users_tasks randomly
            self.users_tasks[i] = CacheMethod.Zipf_sample(task_set=self.task, num=self.each_user_task_n)
        cache_flag = np.zeros(shape=(self.edge_n, self.user_n))
        '''
        cache predict
        '''
        if self.action_wrapper(action) != 0:
            ac_num, ac_task = self.action_wrapper(action)
            for i in range(self.edge_n):
                self.edge_caching_task[i][int(ac_num[i])] = ac_task[i]
        '''
        FIFO
        '''
        # temp = self.edge_caching_task[:, 1:]
        # new_task = np.random.choice(self.task, self.edge_n)
        # self.edge_caching_task = np.column_stack((temp, new_task.T))

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
