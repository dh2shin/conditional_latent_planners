import gym
import numpy as np

import collections
import pickle

import d4rl
import os

for env_name in ['pen', 'hammer', 'door', 'relocate']:
	for dataset_type in ['human', 'expert']:
		name = f'{env_name}-{dataset_type}-v1'
		env = gym.make(name)
		dataset = env.get_dataset()

		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		use_timeouts = False
		if 'timeouts' in dataset:
			use_timeouts = True

		episode_step = 0
		paths = []
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				final_timestep = (episode_step == env._max_episode_steps - 1)
			for k in ['observations', 'actions', 'terminals', 'infos/qpos', 'infos/qvel', 'rewards']:
				data_[k].append(dataset[k][i])
			if env_name == "pen":
				data_["infos/desired_orien"].append(dataset["infos/desired_orien"][i])
			if env_name == "hammer":
				data_["infos/board_pos"].append(dataset["infos/board_pos"][i])
				data_["infos/target_pos"].append(dataset["infos/target_pos"][i])
			if env_name == "door":
				data_["infos/door_body_pos"].append(dataset["infos/door_body_pos"][i])
			# if np.linalg.norm(data_['observations'][-1][:2] - data_['infos/goal'][-1]) <= 0.5:
			# 	data_['rewards'].append(1.0)
			# else:
			# 	data_['rewards'].append(0)
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1

		returns = np.array([np.sum(p['rewards']) for p in paths])
		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		print(f'Number of samples collected: {num_samples}')
		print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		with open(f'./adroit/{name}.pkl', 'wb') as f:
			pickle.dump(paths, f)
