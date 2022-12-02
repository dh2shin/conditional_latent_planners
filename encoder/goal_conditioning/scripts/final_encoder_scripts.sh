#!/bin/bash
"""Training the encoder
"""
size=('umaze' 'medium' 'large')
datasets=('maze2d' 'door' 'pen' 'relocate' 'hammer')
path2repo='path/2/repo'

for dataset in ${datasets[@]}; do
    for size in ${sizes[@]}; do
        expt='brownian_bridge8_${dataset}'
        cd ${path2repo}; python scripts/train_encoder.py --config-name='brownian_bridge' wandb_settings.exp_name=${expt} wandb_settings.exp_dir=${expt} data_params.name=${dataset} data_params.size=${size} model_params.latent_dim=8
    done
done

size=('expert' 'human')
datasets=('door' 'pen' 'relocate' 'hammer')
path2repo='path/2/repo'

for dataset in ${datasets[@]}; do
    for size in ${sizes[@]}; do
        expt='brownian_bridge8_${dataset}'
        cd ${path2repo}; python scripts/train_encoder.py --config-name='brownian_bridge' wandb_settings.exp_name=${expt} wandb_settings.exp_dir=${expt} data_params.name=${dataset} data_params.size=${size} model_params.latent_dim=8
    done
done
