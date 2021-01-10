import gym
import numpy as np
import random
from gym import error, spaces, utils
from gym.utils import seeding

class CacheEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.task_n = 10  # the number of total tasks
    self.task = list(range(1, self.task_n + 1))  # the total tasks set

    self.user_n = 5  # the number of users
    self.each_user_task_n = 1  # the number of requested task by each user
    self.users_tasks = np.zeros(shape=(self.user_n, self.each_user_task_n))
    for i in range(self.user_n):  # initial users_tasks randomly
      self.users_tasks[i] = self.Zipf_sample(self.task, self.each_user_task_n)

    self.edge_n = 3  # the number of agent(edge server)
    self.each_edge_cache_n = 3  # the number of caching tasks by each edge server
    # state : the set of each edge servers' caching tasks
    self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
    self.state = self.edge_caching_task

    # channel gain from edge server to user ！1.5不准确，可更换
    self.h_eu = np.zeros(shape=(self.edge_n, self.user_n))
    for i in range(self.edge_n):
      self.h_eu[i] = 1.5 * abs(1 / np.sqrt(2) * (np.random.randn(self.user_n) + 1j * np.random.randn(self.user_n)))
    # self.h_eu = np.zeros(self.user_n)
    # self.h_eu = 1.5 * abs(1 / np.sqrt(2) * (np.random.randn(self.user_n) + 1j * np.random.randn(self.user_n)))

    # channel gain from cloud server to user
    self.h_cu = abs(1 / np.sqrt(2) * (np.random.randn(self.user_n) + 1j * np.random.randn(self.user_n)))

    # power bisection
    self.p_total = 19.953  # total power is 19.953w, namely 43dbm
    self.p = self.p_total/self.user_n
    # channel bandwidth
    self.bandwidth = 4500000  # 4.5MHZ

    self.seed()  # TODO control some features change randomly
    self.viewer = None  # whether open the visual tool
    self.steps_beyond_done = None  # current step = 0
    self.cache_task_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))

  @property
  def observation_space(self):
    return [spaces.Box(low=0, high=self.task_n, shape=(1, self.edge_n*self.each_edge_cache_n)) for i in range(self.edge_n)]

  @property
  def action_space(self):
    return [spaces.Discrete(np.power(2, self.user_n)) for i in range(self.edge_n)]

  @property
  def agents(self):
    return ['agent' for i in range(self.edge_n)]

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
      hxp = np.power(abs(h[i]*x[i]*self.p), 2)
      sum_hxp += hxp
    for i in range(self.user_n):
      sinr[i] = np.power(abs(h[i]*x[i]*self.p), 2)/(sum_hxp - np.power(abs(h[i]*x[i]*self.p), 2) + self.compute_noise(self.user_n))
    return sinr
  '''
  compute downlink rate
  sinr: SINR from an edge server to all users with the shape of (1, self.user_n)
  '''
  def compute_Rate(self, sinr):
    rate = np.zeros(self.user_n)
    for i in range(self.user_n):
      rate[i] = self.bandwidth * np.log2(1+sinr[i])
    return rate

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  # sample randomly from task_set
  def random_sample(self, task_set, num):
    sample_task = random.sample(task_set, num)
    return sample_task
  '''
  sample a task according to Zipf distribution
  task_set: the set we sample from
  num: the number of sampled tasks
  '''
  def Zipf_sample(self, task_set, num):
    # probability distribution
    task_num = len(task_set)
    p = np.zeros(task_num)
    for i in range(task_num):
      p[i] = int(0.1/(i+1)*100000)
    sampled_task = []
    for j in range(num):
      # sample & return index
      start = 0
      index = 0
      randnum = random.randint(1, sum(p))
      for index, scope in enumerate(p):
        start += scope
        if randnum <= start:
          break
      sampled_task.append(index)
    return sampled_task

  '''
  env_wrapper
  '''
  def action_wrapper(self, acs):
    acs = acs[0]
    ac_str = []
    ac = np.zeros(shape=(len(acs), 5))
    for i in range(len(acs)):
      ac_str.append(bin(list(acs[i]).index(1)).lstrip('0b'))
      for j in range(len(ac_str[i])):
        ac[i][4 - j] = int(ac_str[i][len(ac_str[i]) - 1 - j])
    return ac

  '''
  action is the actions of all agents
  if user_task i in edge_cache j and respng action == 1, serve successfully and mark the user_task
  '''
  def step(self, action):
    action = self.action_wrapper(action)
    edge_caching_task = self.state
    users_tasks = self.users_tasks
    serve_success = np.zeros(shape=(self.edge_n, self.user_n))
    cache_task_flag = self.cache_task_flag
    cache_success_no_action = np.zeros(self.edge_n)
    # now set action as a matrix with the shape of edge_n * user_n
    #ac = np.zeros(self.user_n)
    for i in range(self.user_n):
      for j in range(self.edge_n):
        ac = action[j]
        if users_tasks[i] in edge_caching_task[j] and ac[i] == 1:
          serve_success[j][i] = 1
          flag = np.argwhere(edge_caching_task[j] == users_tasks[i])
          cache_task_flag[j][flag] = 1
        elif users_tasks[i] not in edge_caching_task[j]:
          # # choose new task according to zipf
          # new_task = self.Zipf_sample(self.task, 1)  # zipf: sample a new task according to Zipf
          for k in range(self.each_edge_cache_n):
            if cache_task_flag[j][k] == 0:
              old_task_index = k
              break
          if 0 not in cache_task_flag[j]:
            old_task_index = random.randint(0, self.each_edge_cache_n - 1)
          # choose new task according to user previous task
          temp = np.delete(edge_caching_task[j], old_task_index)  # delete the first one
          edge_caching_task[j] = np.append(temp, users_tasks[i])  # add the new one at the end
        else:
          cache_success_no_action[j] = cache_success_no_action[j] + 1

    # print("users_tasks:\n", users_tasks)
    # print("edge_caching_task:\n", edge_caching_task)
    # print("action:\n", action)
    # print("serve_success:\n", serve_success)
    self.state = edge_caching_task
    self.cache_task_flag = cache_task_flag

    # cloud server serves users successfully
    serve_sucs_cu = np.zeros(self.user_n)
    for i in range(self.user_n):
      if 1 in serve_success[:, i]:
        serve_sucs_cu[i] = 0
      else:
        serve_sucs_cu[i] = 1
    # done
    done = 1

    '''
    initial sinr, downlink rate
      sinr_eu: SINR from edge server to user
      R_eu: downlink rate from edge server to user
      sinr_cu: SINR from cloud server to user
      R_cu: downlink rate from cloud server to user
    compute sinr, downlink rate
    '''
    sinr_eu = np.zeros(shape=(self.edge_n, self.user_n))
    R_eu = np.zeros(shape=(self.edge_n, self.user_n))
    sinr_cu = np.zeros(self.user_n)
    R_cu = np.zeros(self.user_n)

    for i in range(self.edge_n):
      sinr_eu[i] = self.compute_SINR(self.h_eu[i], serve_success[i])
      R_eu[i] = self.compute_Rate(sinr_eu[i])

    sinr_cu = self.compute_SINR(self.h_cu, serve_sucs_cu)
    R_cu = self.compute_Rate(sinr_cu)

    #sinr = int(sum(sum(R_eu)) + sum(R_cu))
    sinr = np.sum(R_eu, axis=1)
    # reward
    #reward = sinr
    reward = 1000*(np.sum(serve_success, axis=1)+1)/(cache_success_no_action+np.sum(serve_success, axis=1)+1)
    # next_obseravtion
    next_obs = self.state
    return next_obs, reward, done, {}

  def reset(self):
    for i in range(self.user_n):  # initial users_tasks randomly
       self.users_tasks[i] = self.Zipf_sample(self.task, self.each_user_task_n)
    # update the state at the beginning of each episode
    for i in range(self.edge_n):
      self.edge_caching_task[i] = self.Zipf_sample(self.task, self.each_edge_cache_n)
    self.state = self.edge_caching_task
    self.steps_beyond_done = None  # set the current step as 0
    self.cache_task_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
    return np.array(self.state)  # return the initial state
  def render(self, mode='human'):
    ...
  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

ACTION_MEANING  = {
  0: "00000",
  1: "00001",
  2: "00010",
  3: "00011",
  4: "00100",
  5: "00101",
  6: "00110",
  7: "00111",
  8: "01000",
  9: "01001",
  10: "01010",
  11: "01011",
  12: "01100",
  13: "01101",
  14: "01110",
  15: "01111",
  16: "10000",
  17: "10001",
  18: "10010",
  19: "10011",
  20: "10100",
  21: "10101",
  22: "10110",
  23: "10111",
  24: "11000",
  25: "11001",
  26: "11010",
  27: "11011",
  28: "11100",
  29: "11101",
  30: "11110",
  31: "11111",

}


