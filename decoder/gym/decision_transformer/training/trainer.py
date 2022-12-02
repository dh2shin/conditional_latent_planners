import numpy as np
import torch
import os

import time


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batches, loss_fn, scheduler=None, eval_fns=None, brownian=False):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batches[0]
        self.get_batch_validation = get_batches[1]
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.best_val_loss = 100
        self.brownian = brownian
        self.eval_frequency = 10


        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False, save_dir=".", dataset=""):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in range(num_steps):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.model.eval()
        if iter_num % self.eval_frequency == 0:
            for eval_fn in self.eval_fns:
                outputs = eval_fn(self.model)
                for k, v in outputs.items():
                    logs[f'evaluation/{k}'] = v
        
        if self.get_batch_validation != None:
            validation_start = time.time()
            self.model.eval()
            validation_losses = []
            for _ in range(int(num_steps/100)):
                validation_loss = self.validation_step()
                validation_losses.append(validation_loss)
            logs['time/validation'] = time.time() - validation_start

        if self.best_val_loss > np.mean(validation_losses):
            print(f"Saving best model iteration No.{iter_num}")
            self.save(directory=save_dir, dataset=dataset, iter_num=iter_num)

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)
        if self.get_batch_validation != None:
            logs['training/validation_loss_mean'] = np.mean(validation_losses)
            logs['training/validation_loss_std'] = np.std(validation_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()

    def validation_step(self):
        return

    def save(self, directory, dataset):
        filename = ""
        t = "brownian" if self.brownian else "base"
        if dataset == "expert" or dataset == "human":
            filename = os.path.join(directory, "{}-{}.pt".format(dataset, t))
        else:
            filename = os.path.join(directory, "{}.pt".format(t))
        torch.save(self.model.state_dict(), filename)