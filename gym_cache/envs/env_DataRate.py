import gym
import numpy as np
import random
from gym import error, spaces, utils
from gym.utils import seeding
from gym_cache.envs.cache_method import CacheMethod

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
    self.state = np.zeros(shape=(self.edge_n, (self.each_edge_cache_n + self.each_edge_cache_n * self.edge_n + self.user_n)))

    self.seed()  # TODO control some features change randomly
    self.viewer = None  # whether open the visual tool
    self.steps_beyond_done = None  # current step = 0
    self.step_num = 0
    
    # power bisection
    self.p_total = 19.953  # total power is 19.953w, namely 43dbm
    self.p = self.p_total/self.user_n
    # channel bandwidth
    self.bandwidth = 4500000  # 4.5MHZ

  @property
  def observation_space(self):
    return [spaces.Box(low=0, high=self.task_n, shape=(self.each_edge_cache_n + self.each_edge_cache_n * self.edge_n + self.user_n, 1)) for i in range(self.edge_n)]

  @property
  def action_space(self):# 选择1个位置，从10个任务中选一个进行缓存更换 + 一种情况不缓存
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
    sum_hxp = 0
    for i in range(self.user_n):
      # hxp = np.power(abs(h[i]*x[i]*self.p), 2)
      hxp = x[i] * np.power(h[i], 2) * self.p
      sum_hxp += hxp
    for i in range(self.user_n):
      # sinr[i] = np.power(abs(h[i]*x[i]*self.p), 2)/(sum_hxp - np.power(abs(h[i]*x[i]*self.p), 2) + self.compute_noise(self.user_n))
      sinr[i] = x[i] * np.power(h[i], 2) * self.p / (
                sum_hxp - x[i] * np.power(h[i], 2) * self.p + np.power(self.compute_noise(self.user_n), 2))
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
  action is the actions of all agents
  if user_task i in edge_cache j and corresponding action == 1, serve successfully and mark the user_task
  cache method 1:
  replace the task no one wants or randomly replace one if each is wanted with a new task wanted by users.
  cache method 2:
  randomly replace one according to Zipf function.
  '''
  def step(self, action):
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
    #reward = [(1-fail/self.user_n)*100/3] * self.edge_n
    reward = np.sum(cache_flag, axis=1)
    print("fail_user:", fail)
    # print("edge_caching:", self.edge_caching_task)

    # change state
    for i in range(self.edge_n):
      self.state[i] = np.append(np.append(self.edge_caching_task[i], np.ndarray.flatten(self.edge_caching_task)), self.users_tasks)

    done = 1
    next_obs = self.state

    return next_obs, reward, done, {}


  def reset(self):
    self.step_num = 0
    #cache_user_fail = np.zeros(self.user_n)
    for i in range(self.user_n):  # initial users_tasks randomly
       self.users_tasks[i] = CacheMethod.Zipf_sample(self.task, self.each_user_task_n)
    # update the state at the beginning of each episode

    self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
    self.state = np.zeros(shape=(self.edge_n, (self.each_edge_cache_n + self.each_edge_cache_n * self.edge_n + self.user_n)))
    self.steps_beyond_done = None  # set the current step as 0

    return np.array(self.state)  # return the initial state
  def render(self, mode='human'):
    ...
  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None
ACTION = {
  0:"012",
  1:"013",
  2:"014",
  3:"023",
  4:"024",
  5:"034",
  6:"123",
  7:"124",
  8:"134",
  9:"234",
}