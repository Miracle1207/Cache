import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
#from maddpg_pytorch.utils.make_env import make_env
from maddpg_pytorch.utils.buffer import ReplayBuffer
#from maddpg_pytorch.utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from maddpg_pytorch.algorithms.maddpg import MADDPG
from gym_cache.envs.cache_env import CacheEnv
from gym_cache.envs.DataRate import DataRate


USE_CUDA = False  # torch.cuda.is_available()

# def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
#     def get_env_fn(rank):
#         def init_env():
#             env = make_env(env_id, discrete_action=discrete_action)
#             env.seed(seed + rank * 1000)
#             np.random.seed(seed + rank * 1000)
#             return env
#         return init_env
#     if n_rollout_threads == 1:
#         return DummyVecEnv([get_env_fn(0)])
#     else:
#         return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    #env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,config.discrete_action)

    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)
    replay_buffer = ReplayBuffer(config.buffer_length, env.edge_n,
                                 [obsp.shape[0] if isinstance(obsp, Box) else obsp.n for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes+config.bi_n_episodes))
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.episode_length):
            env.render()
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[i])), requires_grad=False) for i in range(env.edge_n)] # maddpg.nagents ---3
            #torch_obs = [Variable(torch.Tensor(obs[:, i]), requires_grad=False) for i in range(env.edge_n)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos, cache_flag = env.step(actions)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)

        for a_i, a_ep_rew in enumerate(ep_rews):
            #print('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        logger.add_scalar('total_mean_episode_rewards', sum(ep_rews), ep_i)
        print('total_mean_episode_rewards', sum(ep_rews), ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    access_flag = cache_flag
    for ep_i in range(config.n_episodes, config.n_episodes+config.bi_n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes+config.bi_n_episodes))

        obs = env.bi_reset(access_way=access_flag)
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        for et_i in range(config.cache_episode_length):
            env.render()

            torch_obs = [Variable(torch.Tensor(np.vstack(obs[i])), requires_grad=False) for i in range(env.edge_n)] # maddpg.nagents ---3
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos, cache_flag = env.bi_step(actions)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        access_flag = DataRate.run_BLA(DR, config.bla_steps, connect_flag=cache_flag, p_init=env.p_init, h=env.h)

        for a_i, a_ep_rew in enumerate(ep_rews):
            #print('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        logger.add_scalar('total_mean_episode_rewards', sum(ep_rews), ep_i)
        print('total_mean_episode_rewards', sum(ep_rews), ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')



    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    env = CacheEnv()
    DR = DataRate()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment", default="20210329_data")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents", default="./model")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int) # walker
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e5), type=int)
    parser.add_argument("--n_episodes", default=2, type=int) # 10000 better
    parser.add_argument("--episode_length", default=100, type=int) # TODO increase the lenghth 25

    parser.add_argument("--bi_n_episodes", default=500, type=int)
    parser.add_argument("--cache_episode_length", default=100, type=int)
    parser.add_argument("--bla_steps", default=50, type=int)

    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default=["MADDPG"]*env.edge_n, type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true', default=False)

    config = parser.parse_args()

    run(config)


