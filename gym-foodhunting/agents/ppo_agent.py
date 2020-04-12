# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_foodhunting

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines import PPO2


# https://note.com/npaka/n/nd144f30c8f5b
import os
import tensorflow as tf
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)


def make_env(env_name, rank, seed):
    def _init():
        env = gym.make(env_name)
        model = PPO2.load('1agent1food.zip', verbose=1)
        # model = PPO2.load('2agent1food.zip', verbose=1)
        def policy(obs):
            action, _state = model.predict(obs)
            return action
        env.setPolicies([ policy for _ in range(env.num_agents) ])
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def learn(env_name, seed, load_path, save_path, tensorboard_log, total_timesteps, n_cpu):
    save_path = env_name if save_path is None else save_path
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=save_path)
    eval_env = make_env(env_name, n_cpu, seed)()
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_path+'/best', log_path=tensorboard_log, eval_freq=1000)
    callback = CallbackList([checkpoint_callback, eval_callback])

    policy = CnnPolicy
    # policy = CnnLstmPolicy
    # policy = CnnLnLstmPolicy
    print(env_name, policy)
    # Run this to enable SubprocVecEnv on Mac OS X.
    # export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    # see https://github.com/rtomayko/shotgun/issues/69#issuecomment-338401331
    env = SubprocVecEnv([make_env(env_name, i, seed) for i in range(n_cpu)])
    if load_path is not None:
        model = PPO2.load(load_path, env, verbose=1, tensorboard_log=tensorboard_log)
    else:
        model = PPO2(policy, env, verbose=1, tensorboard_log=tensorboard_log)
    model.learn(total_timesteps=total_timesteps, log_interval=5, callback=callback)
    print('saving model:', save_path+'/latest_model')
    model.save(save_path+'/latest_model')
    env.close()

def play(env_name, seed, load_path, total_timesteps, n_cpu):
    np.set_printoptions(precision=5)
    def padding_obss(obss, dummy_obss):
        dummy_obss[ 0, :, :, : ] = obss
        return dummy_obss
    # trained LSTM model cannot change number of env.
    # so it needs to reshape observation by padding dummy data.
    # dummy_obss = np.zeros((n_cpu, 64, 64, 4))
    # env = SubprocVecEnv([make_env(env_name, 0, seed)])
    env = DummyVecEnv([make_env(env_name, 0, seed)])
    model = PPO2.load(load_path, verbose=1)
    obss = env.reset()
    # obss = padding_obss(obss, dummy_obss)
    rewards_buf = []
    steps_buf = []
    rewards_all_buf = []
    for i in range(total_timesteps):
        actions, _states = model.predict(obss)
        # actions = actions[0:1]
        obss, rewards, dones, infos = env.step(actions)
        # obss = padding_obss(obss, dummy_obss)
        # env.render() # dummy
        if dones[0]:
            steps_buf.append(infos[0]['episode']['l'])
            rewards_all_buf.append(infos[0]['episode']['r_all'])
            r_all = np.array(rewards_all_buf)
            r_mean = [ np.mean(r_all[:,i]) for i in range(r_all.shape[1]) ]
            r_std = [ np.std(r_all[:,i]) for i in range(r_all.shape[1]) ]
            line = np.array([ np.mean(steps_buf), np.std(steps_buf) ] + r_mean + r_std)
            print(len(steps_buf), line)
            # obss = env.reset()
            # obss = padding_obss(obss, dummy_obss)
            if len(steps_buf) % 10 == 0:
                plt.hist(steps_buf, range=(0, 50))
                plt.pause(0.05)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--play', action='store_true', help='play or learn.')
    parser.add_argument('--env_name', type=str, default='FoodHunting-v0', help='environment name.')
    parser.add_argument('--load_path', type=str, default=None, help='filename to load model.')
    parser.add_argument('--save_path', type=str, default=None, help='filename to save model.')
    parser.add_argument('--tensorboard_log', type=str, default=None, help='tensorboard log file.')
    parser.add_argument('--total_timesteps', type=int, default=10000000, help='total timesteps.')
    parser.add_argument('--n_cpu', type=int, default=16, help='number of CPU cores.')
    parser.add_argument('--seed', type=int, default=0, help='seed for random number.')
    args = parser.parse_args()

    if args.play:
        play(args.env_name, args.seed, args.load_path, args.total_timesteps, args.n_cpu)
    else:
        learn(args.env_name, args.seed, args.load_path, args.save_path, args.tensorboard_log, args.total_timesteps, args.n_cpu)
