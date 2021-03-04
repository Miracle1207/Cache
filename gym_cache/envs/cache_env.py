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

        self.user_n = 5  # the number of users
        self.each_user_task_n = 1  # the number of requested task by each user
        self.users_tasks = np.zeros(shape=(self.user_n, self.each_user_task_n))
        for i in range(self.user_n):  # initial users_tasks randomly
            self.users_tasks[i] = CacheMethod.Zipf_sample(task_set=self.task, num=self.each_user_task_n)
        self.edge_n = 3  # the number of agent(edge server)
        self.each_edge_cache_n = 3  # the number of caching tasks by each edge server
        # state : the set of each edge servers' caching tasks
        self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))

        # state = serve_success+flag   agent         user_task     自己缓存的内容的大小
        self.state = np.zeros(
            shape=(self.edge_n, (self.each_edge_cache_n + self.each_edge_cache_n * self.edge_n + self.user_n)))

        self.seed()  # TODO control some features change randomly
        self.viewer = None  # whether open the visual tool
        self.steps_beyond_done = None  # current step = 0
        self.step_num = 0
        # bla
        self.trans_t = 5
        self.cache_t = 10
        self.buffer_t = 1
        self.alpha_1 = np.ones(shape=(self.edge_n, self.each_edge_cache_n))
        self.beta_1 = np.ones(shape=(self.edge_n, self.each_edge_cache_n))
        self.alpha_2 = np.ones(shape=(self.edge_n, self.each_edge_cache_n))
        self.beta_2 = np.ones(shape=(self.edge_n, self.each_edge_cache_n))
        self.cache_buffer_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.change_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.last_25_cache_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.last_25_change_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        # PPP
        self.x1 = 0
        self.y1 = 0
        self.x2, self.y2 = 1, 0
        self.x3, self.y3 = 1, 1.5
        self.lambda0 = 5
        # power bisection
        self.p_total = 19.953  # total power is 19.953w, namely 43dbm
        self.p = self.p_total / self.user_n
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

    def compute_SINR(self, h, x):
        sinr = np.zeros(self.user_n)
        sum_before = 0
        sum_after = 0
        epsilon = 0.001
        for i in range(self.user_n):
            for j in range(i):
                sum_before += x[j] * np.power(h[j], 2) * self.p
            for j in range(i + 1, self.user_n):
                sum_after += x[j] * np.power(h[j], 2) * self.p
            sinr[i] = x[i] * np.power(h[i], 2) * self.p / (
                    sum_before + epsilon * sum_after + np.power(self.compute_noise(1), 2))
        return sinr

    '''
  compute downlink rate
  sinr: SINR from an edge server to all users with the shape of (1, self.user_n)
  '''

    def compute_Rate(self, sinr):
        rate = np.zeros(self.user_n)
        for i in range(self.user_n):
            rate[i] = self.bandwidth * np.log2(1 + sinr[i])
        return rate

    '''
    add BLA:
    beta_compute: generate probability of action
    reward_compute: compute the delay of action
    '''
    # 从beta分布中生成俩值
    def beta_compute(self, a_1, b_1, a_2, b_2):
        temp = 0
        for i in range(int(a_2), int(a_2 + b_2)):
            x_1 = Decimal(math.factorial(i))
            x_2 = Decimal(math.factorial(a_2 + b_2 - 1 - i))
            x_3 = Decimal(math.factorial(a_1 + i - 1))
            x_4 = Decimal(math.factorial(a_2 + b_2 + b_1 - i - 2))
            temp += (x_3 * x_4) / (x_1 * x_2)
        x_5 = Decimal(math.factorial(a_1 + b_1 - 1))
        x_6 = Decimal(math.factorial(a_2 + b_2 - 1))
        x_7 = Decimal(math.factorial(a_1 - 1))
        x_8 = Decimal(math.factorial(b_1 - 1))
        x_9 = Decimal(math.factorial(a_1 + b_1 + a_2 + b_2 - 2))
        p = ((x_5 * x_6) / ( x_7 * x_8 * x_9)) * temp
        return p

    def reward_compute(self, action, change_flag):
        delay = self.trans_t # t = 5
        if action == 0: # cache
            if change_flag == 1:
                delay += self.cache_t # t = 10
        else: # buffer
            if change_flag == 1: # t = 1
                delay += self.buffer_t
        return delay
    '''
    PPP: poisson point process
    user_ppp: input the coordinate(xx0, yy0) of edge server, 
                generate 5 user around it and draw the area of working
                lambda0: the intensity (ie mean density) of the Poisson process
    '''
    def user_ppp(self, xx0, yy0, lambda0):
        # Simulation window parameters
        r = 1  # radius of disk
        areaTotal = np.pi * r ** 2  # area of disk

        # Point process parameters
        numbPoints = np.random.poisson(lambda0 * areaTotal)  # Poisson number of points
        theta = 2 * np.pi * np.random.uniform(0, 1, numbPoints)  # angular coordinates
        rho = r * np.sqrt(np.random.uniform(0, 1, numbPoints))  # radial coordinates
        # Convert from polar to Cartesian coordinates
        xx = rho * np.cos(theta)
        yy = rho * np.sin(theta)
        # Shift centre of disk to (xx0,yy0)
        xx = xx + xx0
        yy = yy + yy0
        return xx, yy

    def draw_edge(self):
        la = 1.0
        fig = plt.figure()
        xx1, yy1 = self.user_ppp(self, 0,0,la)
        xx2, yy2 = self.user_ppp(self, 1,0,la)
        xx3, yy3 = self.user_ppp(self, 1,1.5,la)
        # Plotting
        plt.scatter(xx1, yy1, edgecolor='b', facecolor='none', alpha=0.5)
        plt.scatter(xx2, yy2, edgecolor='g', facecolor='none', alpha=0.5)
        plt.scatter(xx3, yy3, edgecolor='r', facecolor='none', alpha=0.5)
        # margin

        ax = fig.add_subplot(111)
        cir1 = Circle(xy=(0.0, 0.0), radius=1, alpha=0.1, color="b")
        cir2 = Circle(xy=(1.0, 0.0), radius=1, alpha=0.1, color="g")
        cir3 = Circle(xy=(1.0, 1.5), radius=1, alpha=0.1, color="r")
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


    def step(self, action):
        last_cache = copy.deepcopy(self.edge_caching_task)
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
        fail = np.sum(sum(cache_flag) == 0)

        reward = np.zeros(self.edge_n)
        for k in range(self.edge_n):
            reward[k] = sum(self.compute_Rate(self.compute_SINR(self.h[k], cache_flag[k])))

        # change state
        for i in range(self.edge_n):
            self.state[i] = np.append(np.append(self.edge_caching_task[i], np.ndarray.flatten(self.edge_caching_task)),
                                      self.users_tasks)
        '''
        BLA
        '''
        # BLA memory
        if self.step_num % 25 == 0:
            self.last_25_change_flag = copy.deepcopy(self.change_flag)
            self.last_25_cache_flag = copy.deepcopy(self.cache_buffer_flag)
        last_change_flag = copy.deepcopy(self.change_flag)
        last_cache_flag = copy.deepcopy(self.cache_buffer_flag)
        self.cache_buffer_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        self.change_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
        for i in range(self.edge_n):
            for j in range(self.each_edge_cache_n):
                # compute action probability
                p_1 = self.beta_compute(self.alpha_1[i][j], self.beta_1[i][j], self.alpha_2[i][j], self.beta_2[i][j])
                p_2 = 1 - p_1
                # choose action
                if self.edge_caching_task[i][j] in last_cache[i]:
                    index = np.argwhere(last_cache[i] == self.edge_caching_task[i][j])[0][0]
                    if self.cache_buffer_flag[i][index] == 0:
                        self.change_flag[i][j] = 0
                else:
                    self.change_flag[i][j] = 1
                    if p_1 > p_2:
                        self.cache_buffer_flag[i][j] = 0
                    else:
                        self.cache_buffer_flag[i][j] = 1
                if self.step_num % 25 == 24:
                    delay = (self.reward_compute(self.last_25_cache_flag[i][j], self.last_25_change_flag[i][j]) -
                          self.reward_compute(self.cache_buffer_flag[i][j], self.change_flag[i][j]))
                else:
                    delay = (self.reward_compute(last_cache_flag[i][j], last_change_flag[i][j]) -
                             self.reward_compute(self.cache_buffer_flag[i][j], self.change_flag[i][j]))

                # update alpha, beta
                if self.cache_buffer_flag[i][j] == 0:
                    if delay > 0:
                        # reward
                        self.alpha_1[i][j] += 1
                    else:
                        # penalty
                        self.beta_1[i][j] += 1
                else:
                    if delay > 0:
                        # reward
                        self.alpha_2[i][j] += 1
                    else:
                        # penalty
                        self.beta_2[i][j] += 1
        done = 1
        next_obs = self.state
        print("----------------------------")
        print("last cache:", last_cache)
        print("now caching:", self.edge_caching_task)
        print("decision:", self.cache_buffer_flag)
        print("delay:", delay)

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
        self.steps_beyond_done = None  # set the current step as 0

        return np.array(self.state)  # return the initial state

    def render(self, mode='human'):
        ...

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


ACTION = {
    0: "012",
    1: "013",
    2: "014",
    3: "023",
    4: "024",
    5: "034",
    6: "123",
    7: "124",
    8: "134",
    9: "234",
}
