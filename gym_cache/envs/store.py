# # RL: cache network
# self.step_num += 1
# for i in range(self.user_n):  # initial users_tasks randomly
#     self.users_tasks[i] = CacheMethod.Zipf_sample(task_set=self.task, num=self.each_user_task_n)
# cache_flag = np.zeros(shape=(self.edge_n, self.user_n))
# if self.action_wrapper(action) != 0:
#     ac_num, ac_task = self.action_wrapper(action)
#     for i in range(self.edge_n):
#         self.edge_caching_task[i][int(ac_num[i])] = ac_task[i]
#
# for i in range(self.edge_n):
#     for j in range(self.user_n):
#         if self.users_tasks[j] in self.edge_caching_task[i]:
#             cache_flag[i][j] = 1
#
# fail = list(sum(cache_flag)).count(0)
# # cache hit rate
# reward = np.zeros(self.edge_n)
# for k in range(self.edge_n):
#     reward[k] = 100/3*(1-fail/self.user_n)
#
# # change state
# for i in range(self.edge_n):
#     self.state[i] = np.append(np.append(self.edge_caching_task[i], np.ndarray.flatten(self.edge_caching_task)),
#                               self.users_tasks)
# if self.step_num == 24:
#     print("edge:\n", cache_flag)
'''
       BLA: compute best response
'''
# # compute all strategies--datarate
# trans_flag = copy.deepcopy(cache_flag)
# cross_users_index = np.where(sum(self.cache_flag) > 1)[0]
# p = self.p * trans_flag / np.sum(self.p * trans_flag) * self.p_total
# rate_16 = np.zeros(2 ** len(cross_users_index))
# for i in range(2 ** len(cross_users_index)):
#     user_t = self.action_10_2(i, len(cross_users_index))
#     trans_flag = copy.deepcopy(cache_flag)
#     for j in range(len(cross_users_index)):
#         user = int(user_t[j])
#         if user == 0:
#             trans_flag[:, cross_users_index[j]] = 0
#             user_p = p[:, cross_users_index[j]]
#             max_index = np.argmax(user_p)
#             trans_flag[max_index][cross_users_index[j]] = 1
#     rate_16[i] = self.compute_DataRate(trans_flag)
# print("rate_16:", int(rate_16))
# print("JT:", 3*self.compute_DataRate(cache_flag))
# print("ST:", 3*rate_16[0])

'''
BLA : to choose optimal choice
'''
# trans_way = np.zeros(len(cross_users_index))
# self.alpha_1 = [1] * len(cross_users_index)
# self.beta_1 = [1] * len(cross_users_index)
# self.alpha_2 = [1] * len(cross_users_index)
# self.beta_2 = [1] * len(cross_users_index)
#
# trans_flag = copy.deepcopy(self.cache_flag)
# p = self.p * trans_flag / np.sum(self.p * trans_flag) * self.p_total
# last_rate = 0
#
# for i in range(len(cross_users_index)):
#     # sample
#     st = np.random.beta(self.alpha_1[i], self.beta_1[i])
#     jt = np.random.beta(self.alpha_2[i], self.beta_2[i])
#     # choose arm
#     if st > jt:
#         # change trans_flag
#         trans_way[i] = 0
#         trans_flag[:, cross_users_index[i]] = 0
#         user_p = p[:, cross_users_index[i]]
#         max_index = np.argmax(user_p)
#         trans_flag[max_index][cross_users_index[i]] = 1
#     else:
#         trans_way[i] = 1
#     rate = self.compute_DataRate(trans_flag)
#     if rate > last_rate:
#         if trans_way[i] == 0:
#             self.alpha_1[i] += 1
#         else:
#             self.alpha_2[i] += 1
#     else:
#         if trans_way[i] == 0:
#             self.beta_1[i] += 1
#         else:
#             self.beta_2[i] += 1
#     last_rate = rate

'''
BLA:embeded in maddpg
'''
# loop
# BLA_steps=20
# trans_flag = copy.deepcopy(cache_flag)
# p = self.p * trans_flag / np.sum(self.p * trans_flag) * self.p_total
# last_rate = self.compute_DataRate(trans_flag)
# for bla_step in range(BLA_steps):
#     for i in range(len(cross_users_index)):
#         # sample
#         st = np.random.beta(self.alpha_1[i], self.beta_1[i])
#         jt = np.random.beta(self.alpha_2[i], self.beta_2[i])
#         # choose arm
#         if st > jt:
#             # change trans_flag
#             trans_way[i] = 0
#             trans_flag[:, cross_users_index[i]] = 0
#             user_p = p[:, cross_users_index[i]]
#             max_index = np.argmax(user_p)
#             trans_flag[max_index][cross_users_index[i]] = 1
#         else:
#             trans_way[i] = 1
#         rate = self.compute_DataRate(trans_flag)
#         if rate > last_rate:
#             if trans_way[i] == 0:
#                 self.alpha_1[i] += 1
#             else:
#                 self.alpha_2[i] += 1
#         else:
#             if trans_way[i] == 0:
#                 self.beta_1[i] += 1
#             else:
#                 self.beta_2[i] += 1
#         print(rate-last_rate)
#         last_rate = rate

'''
old BLA for buffer or cache
'''
# if self.step_num % 25 == 0:
#     self.last_25_change_flag = copy.deepcopy(self.change_flag)
#     self.last_25_cache_flag = copy.deepcopy(self.cache_buffer_flag)
# last_change_flag = copy.deepcopy(self.change_flag)
# last_cache_flag = copy.deepcopy(self.cache_buffer_flag)
# self.cache_buffer_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
# self.change_flag = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
# for i in range(self.edge_n):
#     for j in range(self.user_n):
#         st = np.random.beta(self.alpha_1, self.beta_1)
#         jt = np.random.beta(self.alpha_2, self.beta_2)
#         if st > jt:
#             arm = 1
#         else:
#             arm = 2
#         if self.step_num % 25 == 24:
#             delay = (self.reward_compute(self.last_25_cache_flag[i][j], self.last_25_change_flag[i][j]) -
#                      self.reward_compute(self.cache_buffer_flag[i][j], self.change_flag[i][j]))
#         else:
#             delay = (self.reward_compute(last_cache_flag[i][j], last_change_flag[i][j]) -
#                      self.reward_compute(self.cache_buffer_flag[i][j], self.change_flag[i][j]))
#
#
#         # update alpha, beta
#         if self.cache_buffer_flag[i][j] == 0:
#             if delay > 0:
#                 # reward
#                 self.alpha_1[i][j] += 1
#             else:
#                 # penalty
#                 self.beta_1[i][j] += 1
#         else:
#             if delay > 0:
#                 # reward
#                 self.alpha_2[i][j] += 1
#             else:
#                 # penalty
#                 self.beta_2[i][j] += 1
#
# print("----------------------------")
# print("last cache:", last_cache)
# print("now caching:", self.edge_caching_task)
# print("decision:", self.cache_buffer_flag)
# print("delay:", delay)

# actions = action[0]
# cache_flag = copy.deepcopy(self.cache_flag[0].reshape(3, 10))
# cross_users_index = np.where(sum(cache_flag) > 1)[0]
# trans_flag = copy.deepcopy(cache_flag)
# p = self.p * trans_flag / np.sum(self.p * trans_flag) * self.p_total
# rate = np.zeros(self.edge_n)
# for i in range(self.edge_n):
#         ac = int(list(actions[i]).index(1))
#         if ac == 0:
#                 # ST
#                 trans_flag[:, cross_users_index[i]] = 0
#                 user_p = p[:, cross_users_index[i]]
#                 max_index = np.argmax(user_p)
#                 trans_flag[max_index][cross_users_index[i]] = 1
#
#         rate[i] = int(self.compute_DataRate(trans_flag))
#
# done = 1
# self.state = np.repeat(trans_flag.reshape(1, 30), 7, axis=0)
#
# next_obs = self.state
#
# reward = rate / self.edge_n
# return next_obs, reward, done, {}