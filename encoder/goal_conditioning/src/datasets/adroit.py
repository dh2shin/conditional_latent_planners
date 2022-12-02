import torch
import random
import os
import json
import pickle
import numpy as np

import torch.utils.data as data
import gym
import d4rl
from goal_conditioning.src.datasets import encoder
from goal_conditioning.src import constants

class AdroitTriplet(data.Dataset):

    def __init__(
            self,
            train,
            seed,
            config,
            manual=None
            ):       
        
        if manual:
            self.env_name = manual['env_name']
            self.size = manual['size']
        else:
            self.env_name = config.data_params.name
            self.size = config.data_params.size
        self.filepath = f"{constants.PATH2ADROIT}{self.env_name}-{self.size}-v1.pkl"
        
        self._load_data()

        super().__init__()

    def _load_data(self):
        with open(self.filepath, 'rb') as f:
            trajectories = pickle.load(f)
            if self.size == "human":
                trajectories = trajectories[:20]
        
        env = gym.make(f'{self.env_name}-{self.size}-v1')
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # From experiment.py in decision transformers
        # save all path information into separate lists
        states, traj_lens, returns = [], [], []
        self.trajectories = []
        for path in trajectories:
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
            self.trajectories.append(path)
        self.traj_lens, returns = np.array(traj_lens), np.array(returns)
        self.start_mean = np.mean([s[0] for s in states], axis=0)
        self.start_std = np.std([s[0] for s in states], axis=0)
        self.end_mean = np.mean([s[-1] for s in states], axis=0)
        self.end_std = np.std([s[-1] for s in states], axis=0)
        print(self.start_std)
        print(self.end_std)


        # used for input normalization
        states = np.concatenate(states, axis=0)


        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        num_timesteps = sum(traj_lens)

        print('=' * 50)
        print(f'Starting new experiment: Adroit {self.size}')
        print(f'{len(self.traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average length: {np.mean(self.traj_lens):.2f}, std: {np.std(self.traj_lens):.2f}')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)

        self.sorted_inds = np.argsort(returns)
        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])

    def __getitem__(self, index):
        inds = index % len(self.traj_lens)
        states, actions, rewards, timesteps, goals = [], [], [], [], []

        traj = self.trajectories[inds]
        timesteps = sorted(np.random.choice(np.arange(traj['rewards'].shape[0]), size=3, replace=True))        
        
        for time in timesteps:
            # get sequences from dataset
            states.append(traj['observations'][time].reshape(1, -1, self.state_dim))
            actions.append(traj['actions'][time].reshape(1, -1, self.act_dim))
            rewards.append(traj['rewards'][time].reshape(1, -1, 1))
        
        y_0 = {
            'states': states[0],
            'actions': actions[0],
            'rewards': rewards[0]
        }
        y_t = {
            'states': states[1],
            'actions': actions[1],
            'rewards': rewards[1]
        }
        y_T = {
            'states': states[2],
            'actions': actions[2],
            'rewards': rewards[2]
        }
        
        result = {
            'y_0': y_0,
            'y_t': y_t,
            'y_T': y_T,
            't_': timesteps[0],
            't': timesteps[1],
            'T': timesteps[2],
            'total_t': traj['rewards'].shape[0],
        }
        return result

    def __len__(self):
        return 100*len(self.traj_lens) if self.size=="human" else 10 * len(self.traj_lens)