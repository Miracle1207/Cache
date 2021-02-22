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
    self.state = np.zeros(shape=(self.edge_n, (1 + self.each_edge_cache_n * self.edge_n + self.user_n)))

    self.seed()  # TODO control some features change randomly
    self.viewer = None  # whether open the visual tool
    self.steps_beyond_done = None  # current step = 0
    self.step_num = 0

  @property
  def observation_space(self):
    return [spaces.Box(low=0, high=self.task_n, shape=(1 + self.each_edge_cache_n * self.edge_n + self.user_n, 1)) for i in range(self.edge_n)]

  @property
  def action_space(self):# 32*(3*10 + 1) 还有一种情况是不换
    return [spaces.Discrete(np.power(2, self.user_n) * (self.each_edge_cache_n * self.task_n + 1)) for i in range(self.edge_n)]

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
    ac_str = []
    serve_ac = np.zeros(shape=(len(acs), self.user_n))
    out_ac = np.zeros(len(acs))
    cache_ac = np.zeros(len(acs))
    for i in range(len(acs)):
      action = list(acs[i]).index(1)
      serve_num = action % np.power(2, self.user_n)
      ac_str.append(bin(serve_num).lstrip('0b'))
      for j in range(len(ac_str[i])):
        serve_ac[i][4 - j] = int(ac_str[i][len(ac_str[i]) - 1 - j])

      if action < np.power(2, self.user_n) * (self.each_edge_cache_n * self.task_n + 1):
        cache_out_num = action // np.power(2, self.user_n)
        out_ac[i] = cache_out_num // self.task_n
        cache_ac[i] = cache_out_num % self.task_n
      else: # 不进行缓存
        out_ac[i] = 3
        cache_ac[i] = 10
    return serve_ac, out_ac, cache_ac

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
    for i in range(self.user_n):  # initial users_tasks randomly
      self.users_tasks[i] = CacheMethod.Zipf_sample(task_set=self.task, num=self.each_user_task_n)
    serve_action, out_action, cache_action = self.action_wrapper(action)
    cache_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
    serve_flag = np.zeros(shape=(self.edge_n, self.user_n))
    edge_cache_flag = np.zeros(shape=(self.edge_n, self.user_n))
    edge_unique = np.zeros(self.edge_n)

    # do action + change flag
    for j in range(self.edge_n):
      out_index = int(out_action[j])
      if out_index < 3 and cache_action[j] < 10:
        self.edge_caching_task[j][out_index] = cache_action[j]
      ac = serve_action[j]
      for i in range(self.user_n):
        if self.users_tasks[i] in self.edge_caching_task[j]:
          edge_cache_flag[j][i] = 1
          if ac[i] == 1:
            serve_flag[j][i] = 1
      for k in range(self.each_edge_cache_n):
        if self.edge_caching_task[j][k] in self.users_tasks:
          cache_flag[j][k] = 1
      edge_unique[j] = len(np.unique(self.edge_caching_task[j]))
    user_cache_flag = np.sum(edge_cache_flag, axis=0)
    user_fail = 0
    for each in user_cache_flag:
      if each == 0:
        user_fail += 1


    # compute reward according to flag              [1,2,3,4,5]
    # total success
    serve_reward = np.sum(serve_flag, axis=1)/self.max_cache_num(self.users_tasks)
    # serve success on condition of caching successfully
    serve_rate = (np.sum(serve_flag, axis=1)+0.001)/(np.sum(edge_cache_flag, axis=1)+0.001)
    # caching tasks be highly used
    cache_reward = np.sum(cache_flag, axis=1)/self.each_edge_cache_n
    # highly cache what users want
    user_reward = 1 - user_fail/self.user_n
    # difference in each agent
    # [1,1,3] [1,3]
    in_agent_dif_rew = edge_unique/self.each_edge_cache_n
    #reward = 100/9*(5*serve_reward+cache_reward+serve_rate+user_reward+in_agent_dif_rew)
    reward = 100* serve_reward
    # state:[1,2,1,3,0]  [1,1,3]
    # action: serve[1 0 1 0 1]  cache [1] in (0,3) ; [3] in (1,10)
    # print
    if self.step_num == 1:
      print("-------------------------------------------------------")
    if self.step_num == 1 or self.step_num == 50 or self.step_num == 99:
      print("step %i:" % self.step_num)
      print("serve_reward:", serve_reward)
      print("serve_rate:", serve_rate)
      print("user_reward:", user_reward)
      print("in_agent_dif_rew:", in_agent_dif_rew)
      print("cache_reward:", cache_reward)
      print("reward: ", reward, '\n\n\n')

    # change state
    for i in range(self.edge_n):
      self.state[i] = np.append(np.append(i, np.ndarray.flatten(self.edge_caching_task)), self.users_tasks)

    done = 1
    next_obs = self.state

    return next_obs, reward, done, {}


  def reset(self):
    self.step_num = 0
    cache_user_fail = np.zeros(self.user_n)
    for i in range(self.user_n):  # initial users_tasks randomly
       self.users_tasks[i] = CacheMethod.Zipf_sample(self.task, self.each_user_task_n)
    # update the state at the beginning of each episode

    self.edge_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
    self.state = np.zeros(shape=(self.edge_n, (1 + self.each_edge_cache_n * self.edge_n + self.user_n)))
    self.steps_beyond_done = None  # set the current step as 0

    return np.array(self.state)  # return the initial state
  def render(self, mode='human'):
    ...
  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None
