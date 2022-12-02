import os
import pickle
import re
from goal_conditioning.src import constants

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb

from goal_conditioning.src.datasets import (
    maze2d,
    adroit
)

import datasets

NAME2DATASET = {
    'maze2d': maze2d.Maze2dTriplet,
    'door': adroit.AdroitTriplet,
    'pen': adroit.AdroitTriplet,
    'relocate': adroit.AdroitTriplet,
    'hammer': adroit.AdroitTriplet,
}

from goal_conditioning.src.models import behavior
from goal_conditioning.src.objectives import brownian_bridge

torch.autograd.set_detect_anomaly(True)

def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=config.optim_params.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        num_workers=config.experiment_params.data_loader_workers,
    )
    return loader

class BrownianBridgeSystem(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._set_dataset()
        self._set_language_encoder()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.optim_params.learning_rate,
            momentum=self.config.optim_params.momentum)
        return [optimizer], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)

    def _set_dataset(self):
        dname = self.config.data_params.name
        dataset = NAME2DATASET[dname]
        self.train_dataset = dataset(
            train=True,
            seed=self.config.data_params.data_seed,
            config=self.config
        )
        self.test_dataset = dataset(
            train=False,
            seed=self.config.data_params.data_seed,
            config=self.config
        )

    def set_to_train(self):
        pass

    def _set_language_encoder(self):
        self.model = behavior.ClapEncoder(latent_dim=self.config.model_params.latent_dim, 
                                        hidden_dim=self.config.model_params.hidden_size, 
                                        size=self.config.data_params.size,
                                        mlp=self.config.model_params.mlp, 
                                        env_name=self.config.data_params.name)

        for p in self.model.model.parameters():
            p.requires_grad = False
        
        print("Trainable params")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)

    def forward(self, states, goals, timesteps):
        feats_state = self.model.forward(states, goals, timesteps)
        return {
            'state': feats_state, 
        }

    def get_feats(self, obs):
        input_ids_i, attention_mask_i = self.train_dataset.tokenize_caption(
            obs, device=self.device)
        input_ids_i = input_ids_i[:, :self.train_dataset.max_length]
        attention_mask_i = attention_mask_i[:, :self.train_dataset.max_length]
        feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        return feats_i

    def get_losses_for_batch(self, batch, batch_idx):
        torch.cuda.empty_cache()
        obs_0 = batch['y_0']
        obs_t = batch['y_t']
        obs_T = batch['y_T']
        t_s = batch['t_'].float()
        ts = batch['t'].float()
        Ts = batch['T'].float()
        if self.config.data_params.name == "maze2d":
            feats_0 = self.forward(obs_0['states'], obs_0['goals'], batch['t_'])
            feats_t = self.forward(obs_t['states'], obs_t['goals'], batch['t'])
            feats_T = self.forward(obs_T['states'], obs_T['goals'], batch['T'])
        else:
            feats_0 = self.forward(obs_0['states'], None, batch['t_'])
            feats_t = self.forward(obs_t['states'], None, batch['t'])
            feats_T = self.forward(obs_T['states'], None, batch['T'])

        loss_fn = brownian_bridge.BrownianBridgeLoss(
            z_0=feats_0['state'].view(feats_0['state'].shape[0], -1),
            z_t=feats_t['state'].view(feats_0['state'].shape[0], -1),
            z_T=feats_T['state'].view(feats_0['state'].shape[0], -1),
            t_=t_s,
            t=ts,
            T=Ts,
            alpha=0,
            var=0,
            loss_type=self.config.loss_params.name,
            eps=self.config.model_params.eps,
            max_seq_len=batch['total_t'].float(),
            pin_start=True if self.config.data_params.name != "maze2d" and self.config.data_params.size == "expert" else False,
            pin_end=True if self.config.data_params.name != "maze2d" and self.config.data_params.size == "expert" else False
        )
        loss = loss_fn.get_loss()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.get_losses_for_batch(batch, batch_idx)
        wandb.log({'train_loss': loss.cpu().detach().numpy(),
                   'epoch': self.trainer.current_epoch})
        self.log('train_loss', loss, prog_bar=True, on_step=True)
        return loss

    def test_step(self, batch, i):
        loss = self.get_losses_for_batch(batch=batch, batch_idx=i)
        wandb.log({'test_loss': loss.cpu().detach().numpy(),
                   'epoch': self.trainer.current_epoch})
        self.log('test_loss', loss, prog_bar=True, on_step=True)
        return loss

    def save(self, directory):
        torch.save(self.model.mlp_state.state_dict(), os.path.join(directory, "mlp_state.pt"))
        # torch.save(self.model.feature_extractor.state_dict(), os.path.join(directory, "feature_extractor.pt"))