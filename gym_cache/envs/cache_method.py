import numpy as np
import random

class CacheMethod(object):
    '''
    sample a task according to Zipf distribution
    task_set: the set we sample from
    num: the number of sampled tasks
    '''
    def Zipf_sample(task_set, num):
        # probability distribution
        task_num = len(task_set)
        p = np.zeros(task_num)
        for i in range(task_num):
            p[i] = int(0.1 / (i + 1) * 100000)
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
    cache method 1: randomly choose one out; choose a new task according to user's requests
    each_edge_cache_n: the number of tasks each edge server wants to cache
    cache_task_flag: if the cache task is needed by users, the flag = 1
    edge_caching_task: edge servers' caching tasks
    users_tasks: the tasks users want
    edge_index: the index of edge server
    user_index: the index of user
    '''
    def random_out_needs_in(self, each_edge_cache_n, cache_task_flag, edge_caching_task, users_tasks, edge_index, user_index):
        for k in range(each_edge_cache_n):
          if cache_task_flag[edge_index][k] == 0:
            old_task_index = k
            break
        if 0 not in cache_task_flag[edge_index]:
          old_task_index = random.randint(0, each_edge_cache_n - 1)
        temp = np.delete(edge_caching_task[edge_index], old_task_index)  # delete the first one
        edge_caching_task[edge_index] = np.append(temp, users_tasks[user_index])  # add the new one at the end
    '''
    cache method 2: randomly choose one out; zipf sample one into cache tasks
    each_edge_cache_n: the number of tasks each edge server wants to cache
    cache_task_flag: if the cache task is needed by users, the flag = 1
    edge_caching_task: edge servers' caching tasks
    task: the task library
    edge_index: the index of edge server
    user_index: the index of user
    '''
    def random_out_zipf_in(self, each_edge_cache_n, edge_caching_task, edge_index, task):
        new_task = self.Zipf_sample(task, 1)
        old_task_index = random.randint(0, each_edge_cache_n - 1)
        temp = np.delete(edge_caching_task[edge_index], old_task_index)
        edge_caching_task[edge_index] = np.append(temp, new_task)
    '''
    cache method 3: set a alpha, if alpha > threshold(alpha_0), randomly replace one, 
    otherwise replace the biggest index(the most unpopular one).
    '''
    def threshold_out_zipf_in(self, each_edge_cache_n, edge_caching_task, task, edge_index):
        new_task = self.Zipf_sample(task, 1)
        alpha = random.random()
        alpha_0 = 1
        if alpha > alpha_0:
          old_task_index = random.randint(0, each_edge_cache_n - 1)
        else:
          old_task_index = np.argmax(edge_caching_task[edge_index])
        temp = np.delete(edge_caching_task[edge_index], old_task_index)
        edge_caching_task[edge_index] = np.append(temp, new_task)
    '''
    cache method 4:
    old_task: choose the max from the tasks no one wants; do not replace if all wanted
    new_task: choose according to zipf function
    '''
    def unpopular_out_zipf_in(self, each_edge_cache_n, cache_task_flag, edge_caching_task, edge_index, task):
        new_task = self.Zipf_sample(task, 1)
        flag_is_0 = []
        for k in range(each_edge_cache_n):
          if cache_task_flag[edge_index][k] == 0:
            flag_is_0.append(k)
        old_task_index = max(flag_is_0, default=0)
        if 0 not in cache_task_flag[edge_index]:
          old_task_index = random.randint(0, each_edge_cache_n - 1)
        temp = np.delete(edge_caching_task[edge_index], old_task_index)  # delete the first one
        edge_caching_task[edge_index] = np.append(temp, new_task)  # add the new one at the end
    '''
    cache method 5: for each agent, randomly choose some tasks from what users request,
    and store in self.agent_i_store_task, each 10 steps, one agent will replace
    one with another from self.agent_i_store_task, which ensures the agent can
    cache all the tasks users want finally.
    '''
    #def steps_update()
    '''
    cache method 6: set cache as an action
    new task: come from cache action
    old task: choose the max from the tasks no one wants; do not replace if all wanted
    '''
    def unpopular_out_cache_action_in(self, each_edge_cache_n, cache_task_flag, edge_caching_task, edge_index, cache_num):
        new_task = cache_num
        flag_is_0 = []
        for k in range(each_edge_cache_n):
            if cache_task_flag[edge_index][k] == 0:
                flag_is_0.append(k)
        old_task_index = max(flag_is_0, default=0)
        if 0 not in cache_task_flag[edge_index]:
            old_task_index = random.randint(0, each_edge_cache_n - 1)
        temp = np.delete(edge_caching_task[edge_index], old_task_index)  # delete the first one
        edge_caching_task[edge_index] = np.append(temp, new_task)  # add the new one at the end
    '''
    cache method 6.5: set cache as an action
    new task: come from cache action
    old task: choose the max from the tasks no one wants; do not replace if all wanted
    '''
    def random_out_cache_action_in(self, each_edge_cache_n, cache_task_flag, edge_caching_task, edge_index, cache_num):
        new_task = cache_num
        flag_is_0 = []
        for k in range(each_edge_cache_n):
            if cache_task_flag[edge_index][k] == 0:
                flag_is_0.append(k)

        if 0 not in cache_task_flag[edge_index]:
            old_task_index = random.randint(0, each_edge_cache_n - 1)
        else:
            old_task_index = random.choice(flag_is_0)

        temp = np.delete(edge_caching_task[edge_index], old_task_index)  # delete the first one
        edge_caching_task[edge_index] = np.append(temp, new_task)  # add the new one at the end
    '''
    cache method 7: cache as an action
    new task: come from cache action
    old task: randomly choose from the tasks no on wants
    '''
    def random_out_action_in(self, cache_num, user_n, cache_success, edge_index, edge_caching_task):
        new_task = cache_num
        flag_is_0 = []
        for k in range(user_n):
            if cache_success[edge_index][k] == 0:
                flag_is_0.append(k)
        if 0 not in cache_success[edge_index]:
            old_task_index = random.randint(0, user_n - 1)
        else:
            old_task_index = random.choice(flag_is_0)
        temp = np.delete(edge_caching_task[edge_index], old_task_index)
        edge_caching_task[edge_index] = np.append(temp, new_task)



