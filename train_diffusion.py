import os
import sys

sys.path.append('.')
sys.path.append('../../preference-diffuser')
from comet_ml import Experiment
import argparse
import pickle
import random
import numpy as np
import torch
import gym
import d4rl

from models.temporal import TrajCondUnet
from models.diffusion import GaussianDiffusion
from data import BlockRankingDataset
from utils.normalizer import DatasetNormalizer
from utils.logger import Logger, make_log_dirs
from utils.trainer import Trainer
from utils.render import Render
from utils.timer import Timer
from utils.evaluator import Evaluator

experiment = Experiment(api_key='ZT2J72DisEsEgG1CPtTK6T5A8', project_name='preference-Diffuser-OPT')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="train-diffusion-opt")
    parser.add_argument("--task", type=str, default="kitchen-mixed-v0")
    parser.add_argument("--domain", type=str, default="gym")
    parser.add_argument("--teacher", type=str, default="block_ranking", choices=["scripted", "human", "block_ranking"])
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--episode-len", type=int, default=450)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--use-human-label", type=bool, default=False)
    # diffusion
    parser.add_argument("--diff-episode-len", type=int, default=450)
    parser.add_argument("--improve-step", type=int, default=20)
    parser.add_argument("--diff-embed-dim", type=int, default=256)
    parser.add_argument("--dim-mults", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--n-diffusion-steps", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--diff-ema-start-epoch", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    parser.add_argument("--diff-lr", type=float, default=5e-5)
    parser.add_argument("--diff-max-iters", type=int, default=500)
    parser.add_argument("--diff-num-steps-per-iter", type=int, default=1000)
    parser.add_argument("--diff-batch-size", type=int, default=32)

    return parser.parse_args()


def train(args=get_args()):
    env = gym.make(args.task)
    dataset = env.get_dataset()

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # load relabeded dataset
    datadir = f'/home/ubantu/LXW/preference-diffuser/data/kitchen-mixed-v0-otr-k1-traj-beta20.0-alpha5.0.pkl'
    with open(datadir, 'rb') as f:
        opt_dataset = pickle.load(f)

    normalizer = DatasetNormalizer(dataset)

    total_return = []
    trajs = []
    for traj in opt_dataset:
        traj_return = 0.0
        traj_dict = {}
        traj_obs = []
        traj_actions = []
        for transition in traj:
            traj_return += transition.reward
            traj_obs.append(transition.observation)
            traj_actions.append(transition.action)
        total_return.append(traj_return)
        traj_dict['observations'] = np.array(traj_obs)
        traj_dict['actions'] = np.array(traj_actions)
        trajs.append(traj_dict)

    total_return = np.array(total_return)
    print('check total num is {}', len(trajs))

    dataset = BlockRankingDataset(dataset, normalizer, total_return, opt_dataset, trajs, device=args.device)
    args.obs_shape = env.observation_space.shape
    args.obs_dim = int(np.prod(args.obs_shape))
    args.action_dim = int(np.prod(env.action_space.shape))
    args.max_action = env.action_space.high[0]

    # crearte diffusion model
    temporal_model = TrajCondUnet(args.diff_episode_len, args.obs_dim + args.action_dim, hidden_dim=args.diff_embed_dim,
                                  dim_mults=args.dim_mults, condition_dropout=args.dropout)
    diffusion_model = GaussianDiffusion(
        model=temporal_model,
        horizon=args.diff_episode_len,
        observation_dim=args.obs_dim,
        action_dim=args.action_dim,
        n_timesteps=args.n_diffusion_steps,
        guidance_scale=args.guidance_scale,
        loss_type='l2',
        clip_denoised=False,
    )
    diffusion_model.to(args.device)
    diffusion_optim = torch.optim.Adam(diffusion_model.parameters(), args.diff_lr)

    # logger
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed)
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "critic_training_progress": "csv",
        "diffusion_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # render
    render = Render(env, args.task, args.obs_dim, args.action_dim)
    # timer
    timer = Timer()
    # evaluator
    evaluator = Evaluator(env, normalizer)

    # trainer
    trainer = OPTTrainer(diffusion_model, dataset, logger, timer, render, evaluator,
                         device=args.device)

    trainer.dataset.block_ranking(args.improve_step)

    print('-------------train diffusion model-------------')
    trainer.train_diffusion(
        optim=diffusion_optim,
        ema_decay=0.995,
        epoch_start_ema=args.diff_ema_start_epoch,
        update_ema_every=10,
        max_iters=args.diff_max_iters,
        num_steps_per_iter=args.diff_num_steps_per_iter,
        batch_size=args.diff_batch_size,
    )


if __name__ == '__main__':
    train()
