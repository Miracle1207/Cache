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
    self.task = list(range(1, self.task_n + 1))  # the total tasks set

    self.user_n = 5  # the number of users
    self.each_user_task_n = 1  # the number of requested task by each user
    self.users_tasks = np.zeros(shape=(self.user_n, self.each_user_task_n))
    for i in range(self.user_n):  # initial users_tasks randomly
      self.users_tasks[i] = CacheMethod.Zipf_sample(task_set=self.task, num=self.each_user_task_n)
    self.edge_n = 3  # the number of agent(edge server)
    self.each_edge_cache_n = 3  # the number of caching tasks by each edge server
    # state : the set of each edge servers' caching tasks
    self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))

    # state = user_index + user_tasks + edge_caching_tasks
    # self.state = np.zeros(shape=(self.edge_n, (self.user_n+self.user_n * self.each_user_task_n+self.edge_n*self.each_edge_cache_n+self.each_edge_cache_n)))
    # self.obs_1 = np.array(range(self.user_n))
    # self.obs_2 = self.users_tasks
    # self.obs_3 = self.edge_caching_task
    # for i in range(self.edge_n):
    #   self.state[i] = np.append(np.append(np.append(self.obs_1, self.obs_2), self.obs_3), self.obs_3[i])
    self.state = self.edge_caching_task
    #self.state = state

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
    self.step_num = 0
    self.fact_total_reward = 0
    '''
    cache method 5 needs:
    '''
    for i in range(self.edge_n):
      self.agent_store_task = self.users_tasks[i:self.each_edge_cache_n+i]

  @property
  def observation_space(self):
    return [spaces.Box(low=0, high=self.task_n, shape=(1, self.edge_n*self.each_edge_cache_n)) for i in range(self.edge_n)]
    # return [spaces.Box(low=0, high=self.task_n, shape=(1, self.user_n + self.each_user_task_n * self.user_n + self.edge_n*self.each_edge_cache_n + self.each_edge_cache_n)) for i in
    #       range(self.edge_n)]
    #return [spaces.Box(low=0, high=self.task_n, shape=(22, 1)) for i in range(self.edge_n)]

  @property
  def action_space(self):
    return [spaces.Discrete(np.power(2, self.user_n)*(self.task_n+1)) for i in range(self.edge_n)]
    #return [i for i in range(np.power(2, self.user_n)*(self.task_n+1))]

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
      #hxp = np.power(abs(h[i]*x[i]*self.p), 2)
      hxp = x[i]*np.power(h[i], 2)*self.p
      sum_hxp += hxp
    for i in range(self.user_n):
      #sinr[i] = np.power(abs(h[i]*x[i]*self.p), 2)/(sum_hxp - np.power(abs(h[i]*x[i]*self.p), 2) + self.compute_noise(self.user_n))
      sinr[i] = x[i]*np.power(h[i], 2)*self.p/(sum_hxp - x[i]*np.power(h[i], 2)*self.p + np.power(self.compute_noise(self.user_n), 2))
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
  one action: action_wrapper
  '''
  # def action_wrapper(self, acs):
  #   acs = acs[0]
  #   ac_str = []
  #   ac = np.zeros(shape=(len(acs), self.user_n))
  #   for i in range(len(acs)):
  #     action = list(acs[i]).index(1)
  #     ac_str.append(bin(action).lstrip('0b'))
  #     for j in range(len(ac_str[i])):
  #       ac[i][4 - j] = int(ac_str[i][len(ac_str[i]) - 1 - j])
  #   return ac
  '''
  two kinds of action: action_wrapper
  '''
  def action_wrapper(self, acs):
    acs = acs[0]
    ac_str = []
    serve_ac = np.zeros(shape=(len(acs), self.user_n))
    cache_ac = np.zeros(self.edge_n)
    for i in range(len(acs)):
      action = list(acs[i]).index(1)
      serve_num = action % np.power(2, self.user_n)
      cache_num = action // np.power(2, self.user_n)
      cache_ac[i] = cache_num
      ac_str.append(bin(serve_num).lstrip('0b'))
      for j in range(len(ac_str[i])):
        serve_ac[i][4 - j] = int(ac_str[i][len(ac_str[i]) - 1 - j])
    return serve_ac, cache_ac
  def max_cache_num(self, user_task):
    uni_len = len(np.unique(user_task))
    if uni_len == 5:
      max_score = 3
    elif uni_len == 4:
      max_score = 4
    else:
      max_score = 5
    return max_score
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
    serve_action, cache_action = self.action_wrapper(action)
    edge_caching_task = self.edge_caching_task
    users_tasks = self.users_tasks
    serve_success = np.zeros(shape=(self.edge_n, self.user_n))
    cache_success = np.zeros(shape=(self.edge_n, self.user_n))
    cache_task_flag = self.cache_task_flag
    cache_success_no_action = np.zeros(self.edge_n)
    #cache_fail = np.zeros(self.edge_n)
    cache_fail = 0

    for j in range(self.edge_n):
      ac = serve_action[j]
      for i in range(self.user_n):
        if users_tasks[i] not in edge_caching_task:
          cache_fail = cache_fail + 1
        if users_tasks[i] in edge_caching_task[j]:
          cache_success[j][i] = 1
          flag = np.argwhere(edge_caching_task[j] == users_tasks[i])
          cache_task_flag[j][flag] = 1
          if ac[i] == 1:
            serve_success[j][i] = 1
          else:
            cache_success_no_action[j] = cache_success_no_action[j] + 1
      CacheMethod.random_out_cache_action_in(CacheMethod, each_edge_cache_n=self.each_edge_cache_n,
                                                  cache_task_flag=self.cache_task_flag,
                                                  edge_caching_task=self.edge_caching_task,
                                                  edge_index=j, cache_num=cache_action[j])
      #CacheMethod.random_out_cache_action_in(CacheMethod, cache_num=cache_action[j], user_n=self.user_n, cache_success=cache_success, edge_index=j, edge_caching_task=self.edge_caching_task)

    # print("users_tasks:\n", users_tasks)
    # print("edge_caching_task:\n", edge_caching_task)
    # print("action:\n", action)
    # print("serve_success:\n", serve_success)

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
    # reward_serve = (np.sum(serve_success, axis=1)+1)/(cache_success_no_action+np.sum(serve_success, axis=1)+1)
    # reward_user_task = 1-cache_fail/(self.edge_n*self.user_n)
    # reward_cache = np.sum(cache_success, axis=1)/self.max_cache_num(users_tasks)
    # #reward_cache = np.sum(cache_task_flag, axis=1)/self.each_edge_cache_n
    # #reward_cache = np.sum(cache_success, axis=1)/(self.user_n * 0.6)
    # reward = (reward_serve + reward_cache + reward_user_task)*(1000/(3*self.edge_n))
    # reward_fact = (reward_serve + reward_cache + reward_user_task)*(1000/(3*self.edge_n))
    # self.fact_total_reward += reward_fact
    reward_final = np.sum(serve_success, axis=1)/self.max_cache_num(users_tasks)
    reward = reward_final*1000/3
    rew = sum(reward)
    if self.step_num == 1 or self.step_num == 50 or self.step_num == 99:
      print("----------------------------------------------\n")
      print("step %i:" %self.step_num)
      print("reward: ", reward)
      # print("reward_serve:", reward_serve)
      # print("reward_user_task:", reward_user_task)
      # print("reward_cache:", reward_cache)
      # print("reward_average:", sum(self.fact_total_reward/self.step_num), "\n\n")


    # state update
    # self.obs_3 = self.edge_caching_task
    # for i in range(self.edge_n):
    #   self.state[i] = np.append(np.append(np.append(self.obs_1, self.obs_2), self.obs_3), self.obs_3[i])
    self.state = self.edge_caching_task
    # # next_obseravtion
    next_obs = self.state
    return next_obs, reward, done, {}

  def reset(self):
    self.step_num = 0
    self.fact_total_reward = 0
    for i in range(self.user_n):  # initial users_tasks randomly
       self.users_tasks[i] = CacheMethod.Zipf_sample(self.task, self.each_user_task_n)
    # update the state at the beginning of each episode
    for i in range(self.edge_n):
      self.edge_caching_task[i] = CacheMethod.Zipf_sample(self.task, self.each_edge_cache_n)

    # self.state = np.zeros(shape=(self.edge_n, (
    #           self.user_n + self.user_n * self.each_user_task_n +
    #           self.edge_n * self.each_edge_cache_n + self.each_edge_cache_n)))
    # self.obs_1 = np.array(range(self.user_n))
    # self.obs_2 = self.users_tasks
    # self.obs_3 = self.edge_caching_task
    # for i in range(self.edge_n):
    #   self.state[i] = np.append(np.append(np.append(self.obs_1, self.obs_2), self.obs_3), self.obs_3[i])

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


