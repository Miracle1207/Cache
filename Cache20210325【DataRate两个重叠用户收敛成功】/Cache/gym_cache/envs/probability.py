import numpy as np

class ProbGenerate(object):
    '''
    The probability matrix of N tasks being transferred to the other tasks
    '''
    def population_generate(self, N, each_num):
        p = np.zeros(shape=(N, N))
        for i in range(N):
            # 圈外, 取N个概率
            p[i] = np.random.uniform(0, 1, N)
            r = i // each_num
            # 圈内, 取each_num个概率
            p[i][each_num * r:each_num * r + each_num] = np.random.uniform(2, 10, each_num)
            c = i + 1
            if c == N:
                c = 0
            # 大概率转移到的那一个
            p[i][c] = np.random.uniform(900, 1000)
            p[i] = p[i] / sum(p[i])
        return p
    '''
    users' new files sample
    '''
    def user_file_sample(self, user_n, file_n, user_file):
        next_file_index = np.zeros(user_n)
        for i in range(user_n):
            file_index = int(user_file[i])
            next_file_index[i] = np.random.choice(range(file_n), p=file_p[file_index])
            # print(file_index, "--", next_file_index[i])
        return next_file_index
# user_n = 20
# file_p = population_generate(file_n, each_num)
# # user file initial
# user_file = np.random.choice(range(file_n), user_n)
# file_n = 100
# each_num = 10
# for step_i in range(100):
#     user_next_file = user_file_sample(user_n, file_n, user_file)
#     print(step_i,":")
#     print("user_file:", user_file)
#     user_file = user_next_file