from gym_cache.envs.cache_env import CacheEnv
import numpy as np

if __name__ == '__main__':
  env = CacheEnv()
  max_iter = 1000
  env.reset()
  for k in range(max_iter):
    print("iter = ", k)
    action_list = np.random.randint(0, 2, (env.edge_n, env.user_n))
    _, reward, _, _ = env.step(action_list)
    print("reward: ", reward)