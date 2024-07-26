import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from comet_ml import Experiment
from copy import deepcopy
from utils.helpers import EMA, cycle, batch_to_device
from data.ranking_data import ActorDataset


use_comet = False
if use_comet:
    experiment = Experiment(api_key='ZT2J72DisEsEgG1CPtTK6T5A8', project_name='preference-trainactor')
def get_tlen(trajs, normalizer, obs_dim):
    tlens = []
    max_len = trajs.shape[1]
    for traj in trajs:
        observations = normalizer.unnormalize(traj[..., :obs_dim], "observations")
        actions = normalizer.unnormalize(traj[..., obs_dim:], "actions")
        for i in range(max_len):
            if np.sum(np.abs(observations[i])) + np.sum(np.abs(actions[i])) < 1e-2:
                tlens.append(i)
                break
            if i == max_len - 1:
                tlens.append(max_len)
    return tlens
         
class BCTrainer:
    def __init__(self, actor, dataset, logger, timer, evaluator, experiment , device='cuda:0') -> None:
        self.actor = actor
        self.dataset = dataset
        self.logger = logger
        self.timer = timer
        self.evaluator = evaluator
        self.device = device
        self.experiment = experiment

    def train(
        self, 
        optim, 
        max_iters, 
        num_steps_per_iter,
        batch_size,

    ):
        self.step = 0
        self.timer.reset()
        for iter in range(max_iters):
            for step in range(num_steps_per_iter):
                observations, actions = self.dataset.sample_batch(batch_size)
                actor_loss = self.actor.loss(observations, actions)
                
                optim.zero_grad()
                actor_loss.backward()
                optim.step()

                self.step += 1
                self.logger.logkv('actor loss', actor_loss.item())
                self.experiment.log_metric('actor_loss', actor_loss ,step=iter * num_steps_per_iter + step)
            
            # log
            elapsed_time, total_time = self.timer.reset()
            elapsed_fps = num_steps_per_iter / elapsed_time
            self.logger.logkv('actor fps', elapsed_fps)
            self.logger.logkv('actor total time', total_time)
            
            for k, v in self.evaluator.evaluate(self.actor).items():
                self.logger.logkv(k, v)
                self.experiment.log_metric(k, v,step=iter)
            self.logger.set_timestep(iter)
            self.logger.dumpkvs(exclude=["actor_training_progress"])
            
        torch.save(self.actor.state_dict(), os.path.join(self.logger.model_dir, f'actor.pth'))


class Trainer:
    def __init__(self, diffusion_model, dataset, logger, timer, render, evaluator, device='cuda:0') -> None:
        self.dataset = dataset
        self.diffusion_model = diffusion_model
        self.logger = logger
        self.timer = timer
        self.render = render
        self.evaluator = evaluator
        self.device = device

    def train_diffusion(
            self,
            optim,
            ema_decay,
            epoch_start_ema,
            update_ema_every,
            max_iters,
            num_steps_per_iter,
            batch_size,
            render_epoch=20,
    ):
        self.ema = EMA(ema_decay)
        self.ema_model = deepcopy(self.diffusion_model)
        self.update_ema_every = update_ema_every
        self.epoch_start_ema = epoch_start_ema
        self.step = 0

        self.diffusion_model.train()
        save_dir = os.path.join(self.logger._video_dir, 'iter=0')
        # self.render_generator(save_dir, mode="train")

        self.timer.reset()
        for iter in range(max_iters):  # 500
            # train step
            for step in range(num_steps_per_iter):  # 1000
                # 19*1000*14
                condition_traj, target_traj, init_obs, distance, _ = self.dataset.sample_batch(batch_size)

                loss, infos = self.diffusion_model.loss(target_traj, condition_traj, init_obs,distance)

                optim.zero_grad()
                loss.backward()
                optim.step()

                self.step += 1
                if use_comet:
                    experiment.log_metric("diffusion(OPT)-train_loss", loss, step=step)
                print("diffusion-train-loss is {}，step is {},itr is {}".format(loss, step, iter))

                for k, v in infos.items():
                    self.logger.logkv_mean(k, v.item())

            elapsed_time, total_time = self.timer.reset()
            elapsed_fps = num_steps_per_iter / elapsed_time
            self.logger.logkv('diffusion fps', elapsed_fps)
            self.logger.logkv('diffusion total time', total_time)

            # render generation
            if (iter + 1) % 5 == 0:  # 20
                model_dir = './lxv_241347-opt/hopper-medium-replay'
                torch.save(self.diffusion_model.state_dict(), os.path.join(model_dir, f'diffusion-%d.pth' % (iter + 1)))

            # log
            self.logger.set_timestep(iter)
            self.logger.dumpkvs(exclude=["diffusion_training_progress"])