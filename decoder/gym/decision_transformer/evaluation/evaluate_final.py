import torch
import sys

from utils import _maze_evaluation_pts, _label_with_episode_number
sys.path.append("../..")
from decision_transformer.models.decision_transformer import DecisionTransformer, DecisionTransformerWithBrownian
import gym
import d4rl
import numpy as np
import pickle
import argparse
import pandas as pd

def evaluate(variant, seed):
    device = variant['device']
    dataset = variant['dataset']
    brownian = variant['brownian']
    iterations = variant['num_eval_episodes']
    video = variant['video']
    
    env = gym.make(f'maze2d-{dataset}-v1')
    max_ep_len = env._max_episode_steps
    env_targets = variant['target_return']  # evaluation conditioning targets
    scale = 1000.  # normalization for rewards/returns
    state_dim = env.observation_space.shape[0] + 2
    act_dim = env.action_space.shape[0]

    #######################################################
    # load dataset
    dataset_path = f'/data/ds816/decision-transformer/gym/data/maze2d-{dataset}-v1-relabeled-correctly.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    s, traj_lens, returns = [], [], []
    new_trajectories = []
    for path in trajectories:
        new_trajectories.append(path)
        s.append(np.concatenate([path['observations'], path['infos/goal']], axis=1))
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    trajectories = new_trajectories
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    start_mean = np.mean([st[0] for st in s], axis=0)
    start_std = np.std([st[0] for st in s], axis=0)
    end_mean = np.mean([st[-1] for st in s], axis=0)
    end_std = np.std([st[-1] for st in s], axis=0)

    # used for input normalization
    s = np.concatenate(s, axis=0)
    state_mean, state_std = np.mean(s, axis=0), np.std(s, axis=0) + 1e-6
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting evaluation on maze2d {dataset}')
    print(f'Dataset stats:')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average trjectory length: {num_timesteps/len(traj_lens):.2f}')
    print(f'Max trajectory length: {max(traj_lens):.2f}')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    #######################################################
    if brownian:
        model = DecisionTransformerWithBrownian(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=20,
            max_ep_len=max_ep_len,
            hidden_size=128,
            augment_type='append',
            n_layer=3,
            n_head=1,
            n_inner=4*128,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            mlp=False,
            traj_len=int(np.mean(traj_lens)),
            start_mean=start_mean, start_std=start_std, end_mean=end_mean, end_std=end_std,
            multigoal=True,
            multistart=True
        )
    else:
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=20,
            max_ep_len=max_ep_len,
            hidden_size=128,
            n_layer=3,
            n_head=1,
            n_inner=4*128,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
        )

    path = variant['path']
    if path != "":
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(_maze_evaluation_pts(dataset, brownian))
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    #######################################################
    TARGET_RETURN = env_targets / scale

    episode_returns, episode_lengths = [], []
    frames = []
    env.seed(seed)
    for ep in range(iterations):
        episode_return, episode_length = 0, 0
        state = env.reset()
        state = np.concatenate([state, env.get_target()])

        if brownian:
            model.latent_plan = model.generate_latent_plan(start=state, goal=env.get_target())

        target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        for t in range(max_ep_len):
            if video:
                if t % 5 == 0:
                    frame = env.render(mode='rgb_array')
                    frames.append(_label_with_episode_number(frame, episode_num=seed*iterations + ep))
            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])

            action = model.get_action(
                (states.to(dtype=torch.float32)) if not variant['normalize'] else (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long)
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = env.step(action)
            
            state = np.concatenate([state, env.get_target()])

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            pred_return = target_return[0, -1] - (reward / scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
    print('=' * 50)
    print(f'Policy Evaluation')
    print(f'Average return: {np.mean(episode_returns):.2f}')
    print('%s Average length (%d ep): %f' % (dataset, iterations, np.mean(episode_lengths)))
    print('=' * 50)
    normalized_scores = []
    for returns in episode_returns:
        normalized_scores.append(env.get_normalized_score(returns))
    print("Normalized score: ", np.mean(normalized_scores))
    return episode_returns, episode_lengths, normalized_scores
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='medium')
    parser.add_argument('--brownian', type=bool, default=False)
    parser.add_argument('--target_return', type=float, default=50)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--video', type=bool, default=False)
    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--normalize', type=bool, default=False)
    
    args = parser.parse_args()

    dataset = vars(args)['dataset']

    episode_returns, episode_lengths, scores = [], [], []
    for seed in range(4):
        episode_return, episode_length, score = evaluate(variant=vars(args), seed=seed)    
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        scores.append(score)
    data = {
        "Dataset": dataset,
        "Normalized Score Mean": np.mean(scores), 
        "Normalized Score Std": np.std(scores), 
    }
    print("Final Metrics:")
    print(data)