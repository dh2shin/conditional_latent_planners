import sys
from clap.decoder.gym.decision_transformer.evaluation.utils import _adroit_evaluation_pts, _maze_evaluation_pts

from goal_conditioning.src.models.decision_transformer import DecisionTransformer
from goal_conditioning.src.models.utils import weights_init
import torch
import torch.nn as nn
import gym
import numpy as np
import d4rl

from goal_conditioning.src.models.utils import weights_init

class ClapEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, size, mlp=False, env_name="maze2d"):
        super(ClapEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.env_name = env_name
        self.size = size
        self.mlp = mlp
        self._init_model()

    def _init_model(self):
        if self.env_name == "maze2d":
            env = gym.make(f'maze2d-{self.size}-v1')
            max_ep_len = env._max_episode_steps
            state_dim = env.observation_space.shape[0] + 2
            act_dim = env.action_space.shape[0]

            self.model = DecisionTransformer(
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
            state_dict = torch.load(_maze_evaluation_pts(dataset=self.size, brownian=False))
            self.model.load_state_dict(state_dict)
        elif self.env_name == "door" or self.env_name == "pen" or self.env_name == "hammer" or self.env_name == "relocate":
            env = gym.make(f'{self.env_name}-{self.size}-v1')
            state_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            state_dict = torch.load(_adroit_evaluation_pts(expert=self.size, dataset=self.env_name, brownian=False))
            max_ep_len = state_dict['embed_timestep.weight'].shape[0]
            self.model = DecisionTransformer(
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
            self.model.load_state_dict(state_dict)
        else:
            raise NotImplementedError
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = self.finetune

        self.feature_extractor = self.create_feature_extractor() # data_dim -> hidden_dim

        if self.mlp:
            self.mlp_state = nn.Linear(self.model.embed_state.weight.shape[0], self.hidden_dim)
        else:
            self.mlp_state = nn.Linear(self.model.embed_state.weight.shape[0], self.latent_dim)

        self.mlp_state.apply(weights_init)
        self.feature_extractor.apply(weights_init)

    def create_feature_extractor(self):
        return nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
                               ])

    def set_to_train(self):
        pass

    def forward(self, states, goals, timesteps):
        if self.env_name == "maze2d":
            states = torch.cat((states, goals), -1)
        
        state_emb = self.model.embed_state(states)
        state_emb = self.mlp_state(state_emb)

        if self.mlp:
            state_emb = self.feature_extractor(state_emb)
        #############ADD TIME EMBEDDING############
        return state_emb
