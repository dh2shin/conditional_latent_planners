from multiprocessing.spawn import prepare

from decision_transformer.evaluation.utils import _adroit_mlp_pts, _adroit_evaluation_pts, _maze_evaluation_pts, _maze_mlp_pts
import gym
import numpy as np
import torch
import wandb
import d4rl

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer, DecisionTransformerWithBrownian
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from constants import _get_env_dataset, _get_env_settings, _get_weights_directory

import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def prepare_dataloader(trajectories, env_name="maze2d", dataset="medium"):
    states, traj_lens, returns = [], [], []
    new_trajectories = []
    for path in trajectories:
        new_trajectories.append(path)
        if env_name == 'maze2d':
            states.append(np.concatenate([path['observations'], path['infos/goal']], axis=1))
        else:
            states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    trajectories = new_trajectories
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    start_mean = np.mean([s[0] for s in states], axis=0)
    start_std = np.std([s[0] for s in states], axis=0)
    end_mean = np.mean([s[-1] for s in states], axis=0)
    end_std = np.std([s[-1] for s in states], axis=0)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    # state_mean, state_std = np.concatenate([np.mean(states, axis=0), [0,0]]), np.concatenate([np.std(states, axis=0) + 1e-6, [1,1]])
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Preparing Dataloader: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    return states, traj_lens, returns, trajectories, state_mean, state_std, start_mean, start_std, end_mean, end_std

def get_batch(batch_size, max_len, num_trajectories, p_sample, trajectories, sorted_inds, env_name, state_dim, act_dim, max_ep_len, scale, device):
    batch_inds = np.random.choice(
        np.arange(num_trajectories),
        size=batch_size,
        replace=True,
        p=p_sample,  # reweights so we sample according to timesteps
    )

    s, a, r, d, rtg, timesteps, mask, next_s = [], [], [], [], [], [], [], []
    for i in range(batch_size):
        traj = trajectories[int(sorted_inds[batch_inds[i]])]
        si = random.randint(0, traj['rewards'].shape[0] - 1)

        # get sequences from dataset
        if env_name == 'maze2d':
            s.append(np.concatenate([traj['observations'][si:si + max_len],traj['infos/goal'][si:si + max_len]], axis=1).reshape(1, -1, state_dim))
        else:
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
        
        # get next state for brownian bridge
        if si + max_len <= traj['rewards'].shape[0] - 1:
            if env_name == "maze2d":
                next_s.append(np.concatenate([traj['observations'][si+max_len], traj['infos/goal'][si + max_len]]).reshape(1, -1, state_dim))
            else:
                next_s.append(traj['observations'][si+max_len].reshape(1, -1, state_dim))
        else:
            if env_name == "maze2d":
                next_s.append(np.concatenate([traj['observations'][-1], traj['infos/goal'][-1]]).reshape(1, -1, state_dim))
            else:
                next_s.append(traj['observations'][-1].reshape(1, -1, state_dim))
            # next_s.append(np.zeros((1, 1, state_dim))) # NOTE(zero-padding): zero padding because dt people do that

        a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
        r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
        if 'terminals' in traj:
            d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
        else:
            d.append(traj['dones'][si:si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
        rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
        if rtg[-1].shape[1] <= s[-1].shape[1]:
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

        # padding and state + reward normalization
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
        a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
        timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
        mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    next_s = torch.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=torch.float32, device=device)
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

    return s, a, r, d, rtg, timesteps, mask, next_s

def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'
    brownian = variant['brownian']

    env = gym.make(f'{env_name}-{dataset}-v1')
    max_ep_len = env._max_episode_steps
    env_targets, scale = _get_env_settings(env_name, dataset)

    env.seed(0)

    state_dim = env.observation_space.shape[0] + 2 if env_name == 'maze2d' else env.observation_space.shape[0] # because of infos/goal
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path, validation_dataset_path = _get_env_dataset(env_name, dataset)
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    with open(validation_dataset_path, 'rb') as f:
        if dataset == "human":
            validation_trajectories = trajectories[-5:]
            trajectories = trajectories[:-5]
        elif dataset == "expert":
            validation_trajectories = pickle.load(f)
            validation_trajectories = validation_trajectories[:100]
        else:
            validation_trajectories = [trajectories[i] for i in range(len(trajectories)) if i % 50 == 0]
            trajectories = [trajectories[i] for i in range(len(trajectories)) if i % 50 != 0]


    # save all path information into separate lists
    states, traj_lens, returns, trajectories, state_mean, state_std, start_mean, start_std, end_mean, end_std = prepare_dataloader(trajectories, env_name=env_name, dataset=dataset)

    if dataset == "human":
        max_ep_len = np.max(traj_lens) # NOTE(human dataset length adjustment)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']

    validation_states, validation_traj_lens, validation_returns, validation_trajectories, validation_state_mean, validation_state_std, _, _, _, _ = prepare_dataloader(validation_trajectories,env_name=env_name,dataset=dataset)
    validation_sorted_inds = np.argsort(validation_returns)  # lowest to highest
    validation_num_trajectories = len(validation_sorted_inds)
    validation_p_sample = validation_traj_lens[validation_sorted_inds] / sum(validation_traj_lens[validation_sorted_inds])
    
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = len(sorted_inds)
    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    mode = variant.get('mode', 'normal')

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            normalized_scores = []
            prev_seed = 0
            env.seed(prev_seed)
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if int(_ / 5) != prev_seed:
                        prev_seed = int(_ / 5)
                        env.seed(prev_seed)
                    ret, length, score = evaluate_episode_rtg(
                        env,
                        env_name,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        brownian=brownian
                    )
                returns.append(ret)
                lengths.append(length)
                normalized_scores.append(score)
            
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_score_mean': np.mean(normalized_scores),
                f'target_{target_rew}_score_std': np.std(normalized_scores),
            }
        return fn

    if brownian:
        model = DecisionTransformerWithBrownian(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            latent_dim=variant['latent_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            mlp=variant['mlp'],
            augment_type='append',
            traj_len=(int(np.mean(traj_lens)) if variant['traj_len'] == -1 else variant['traj_len']),
            multigoal=True,
            start_mean=start_mean, start_std=start_std, end_mean=end_mean, end_std=end_std
        )
        
        if env_name == "maze2d":
            state_dict = torch.load(_maze_evaluation_pts(dataset, False))
            mlp_state_dict = torch.load(_maze_mlp_pts(dataset, False)) 
        else:
            state_dict = torch.load(_adroit_evaluation_pts(dataset == "expert", dataset, False))
            mlp_state_dict = torch.load(_adroit_mlp_pts(dataset == "expert", dataset, False)) 

        copy_state_dict = state_dict
        for k,v in mlp_state_dict.items():
            new_k = "mlp_state." + k
            copy_state_dict[new_k] = v
        
        # if variant['mlp']:
        #     feature_extractor_dict = torch.load(f'/data/ds816/decision-transformer/gym/encoder_pts/{dataset}/feature_extractor.pt')
        #     for k,v in feature_extractor_dict.items():
        #         new_k = "feature_extractor." + k
        #         copy_state_dict[new_k] = v
        model.load_state_dict(copy_state_dict, strict=False)
    else:
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batches=[get_batch(256, K, num_trajectories, traj_lens[sorted_inds] / sum(traj_lens[sorted_inds]), trajectories, sorted_inds, env_name, state_dim, act_dim, max_ep_len, scale, device), 
                    get_batch(256, K, validation_num_trajectories, validation_p_sample, validation_trajectories, validation_sorted_inds, env_name, state_dim, act_dim, max_ep_len, scale, device)],
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
        brownian=brownian
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,entity="dh2shin",
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    save_dir = _get_weights_directory(env_name, dataset)
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True, save_dir=save_dir, dataset=dataset)
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='maze2d')
    parser.add_argument('--dataset', type=str, default='expert')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--brownian', type=bool, default=False)
    parser.add_argument('--mlp', type=bool, default=False)
    parser.add_argument('--traj_len', type=int, default=-1)
    parser.add_argument('--video', type=bool, default=False)
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('--latent_dim', type=int, default=8)
    
    
    args = parser.parse_args()

    experiment('gym-experiment' if vars(args)['exp_name'] == "" else vars(args)['exp_name'], variant=vars(args))
