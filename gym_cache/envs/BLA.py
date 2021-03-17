import numpy as np
import random
import math
import gym

from gym_cache.envs.cache_method import CacheMethod

# initialize
# state = (own caching contents) all are cached
# action: for each caching content, choose to cache or buffer (0,1)
# reward: total delay in one episode; for each step, reward = caching time + transimission time + buffer time + transmission time



# 从beta分布中生成俩值
def beta_compute(a_1, b_1, a_2, b_2):
    temp = 0
    for i in range(int(a_2), int(a_2+b_2)):
        temp += (math.factorial(a_1+i-1)*math.factorial(a_2+b_2+b_1-i-2))/(math.factorial(i)*math.factorial(a_2+b_2-1-i))
    p = (math.factorial(a_1+b_1-1)*math.factorial(a_2+b_2-1))/(math.factorial(a_1-1)*math.factorial(b_1-1)*math.factorial(a_1+b_1+a_2+b_2-2))*temp
    return p
def reward_compute(action, change_flag):
    delay = trans_t
    if action == 0:
        if change_flag == 1:
            delay += cache_t
    else:
        if change_flag == 1:
            delay += buffer_t
    return delay

# 比较俩值，根据大小选择其中一个action
# 更新参数（BLA参数，state，计算reward）
def step(last_caching, next_caching):
    delay = np.zeros(shape=(3, cache_space))
    for j in range(3):
        for k in range(cache_space):
            # select action according to BLA
            p_1 = beta_compute(alpha_1[j][k], beta_1[j][k], alpha_2[j][k], beta_2[j][k])
            p_2 = beta_compute(alpha_2[j][k], beta_2[j][k], alpha_1[j][k], beta_1[j][k])

            # compute cache reward
            # 现在先设计若存在上次缓存中，可以暂且设定 buffer和cache可以互换
            if next_caching[j][k] in last_caching[j]:
                flag = 0
            else:
                flag = 1
            cache_delay = reward_compute(0, flag)
            buffer_delay = reward_compute(1, 1)

            if p_1 > p_2:
                # cache  0
                if action[j][k] == 1:
                    change_flag[j][k] = 1
                    action[j][k] = 0
                else:
                    change_flag[j][k] = 0
            else:
                # buffer  1
                if action[j][k] == 0:
                    change_flag[j][k] = 1
                    action[j][k] = 1
                else:
                    # 没改变
                    change_flag[j][k] = 0
            # 设计reward or penalty + 更新参数
            if action[j][k] == 0:
                delay[j][k] = cache_delay
                if cache_delay < buffer_delay:
                    alpha_1[j][k] += 1
                else:
                    beta_1[j][k] += 1
            else:
                delay[j][k] = buffer_delay
                if cache_delay > buffer_delay:
                    alpha_2[j][k] += 1
                else:
                    beta_2[j][k] += 1
            print("p1:", p_1)
            print(("p2:", p_2))
    last_caching = next_caching
    return delay, last_caching
    # 这个reward是在分完buffer,cache之后对应的一个step中的每个agent 每个缓存内容对应的delay，需要累加用一个episode的reward与之前比较



task = list(range(10))
user_n = 5
each_user_task_n = 1
action_space = 2

trans_t = 5
cache_t = 10
buffer_t = 1

episodes_n = 1
steps_n = 25

agents_n = 3
cache_space = 3
last_caching = np.zeros(shape=(3, 3))
next_caching = np.zeros(shape=(agents_n, cache_space))
users_tasks = np.zeros(shape=(user_n, each_user_task_n))

alpha_1 = np.ones(shape=(agents_n, cache_space))
beta_1 = np.ones(shape=(agents_n, cache_space))
alpha_2 = np.ones(shape=(agents_n, cache_space))
beta_2 = np.ones(shape=(agents_n, cache_space))
action = np.zeros(shape=(agents_n, cache_space))
change_flag = np.ones(shape=(agents_n, cache_space))
total_delay = np.zeros(3)
for step_i in range(1000):
    for i in range(5):  # initial users_tasks randomly
        users_tasks[i] = CacheMethod.Zipf_sample(task_set=task, num=1)
    users_tasks[i] = CacheMethod.Zipf_sample(task_set=task, num=1)
    e = random.random()
    for i in range(agents_n):
        for j in range(cache_space):
            if e<0.7:
                next_caching[i][j] = random.choice(users_tasks)
            else:
                next_caching[i][j] = random.choice(task)


    step_delay, last_caching = step(last_caching, next_caching)
    step_delay = np.sum(step_delay, axis=1)
    print(step_i, ":", step_delay)
    total_delay += step_delay

print("total_delay:", total_delay)




