# import d4rl
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions.normal as Normal
import ipdb
import random
import pickle


def reparameterize(mean, std):
    eps = torch.normal(torch.zeros(mean.size()).cuda(), torch.ones(mean.size()).cuda())
    return mean + std * eps


def stable_weighted_log_sum_exp(x, w, sum_dim):
    a = torch.min(x)
    ipdb.set_trace()

    weighted_sum = torch.sum(w * torch.exp(x - a), sum_dim)

    return a + torch.log(weighted_sum)


def chunks(obs, actions, H, stride):
    '''
    obs is a N x 4 array
    goals is a N x 2 array
    H is length of chunck
    stride is how far we move between chunks.  So if stride=H, chunks are non-overlapping.  If stride < H, they overlap
    '''

    obs_chunks = []
    action_chunks = []
    N = obs.shape[0]
    for i in range(N // stride - H):
        start_ind = i * stride
        end_ind = start_ind + H

        obs_chunk = torch.tensor(obs[start_ind:end_ind, :], dtype=torch.float32)

        action_chunk = torch.tensor(actions[start_ind:end_ind, :], dtype=torch.float32)

        loc_deltas = obs_chunk[1:, :2] - obs_chunk[:-1, :2]  # Franka or Maze2d

        norms = np.linalg.norm(loc_deltas, axis=-1)
        # USE VALUE FOR THRESHOLD CONDITION BASED ON ENVIRONMENT
        if np.all(
                norms <= 0.8):  # Antmaze large 0.8 medium 0.67 / Franka 0.23 mixed/complete 0.25 partial / Maze2d 0.22
            obs_chunks.append(obs_chunk)
            action_chunks.append(action_chunk)
        else:
            pass

    print('len(obs_chunks): ', len(obs_chunks))
    print('len(action_chunks): ', len(action_chunks))

    return torch.stack(obs_chunks), torch.stack(action_chunks)


def get_dataset(env_name, stride=20, horizon=100, test_split=0.2, isnormalize=True):
    dataset_file = '../data/' + env_name + '.pkl'
    with open(dataset_file, "rb") as f:
        dataset = pickle.load(f)

    observations = []
    actions = []


    obs = dataset['observations']
    act = dataset['actions']

    if isnormalize:
        obs_mean = np.mean(obs,axis=0)
        act_mean = np.mean(act,axis=0)
        eps = 1e-3
        obs_std = np.std(obs,axis=0) + eps
        act_std = np.std(act,axis=0) + eps
        obs = (obs - obs_mean) / obs_std
        act = (act - act_mean) / act_std


    # rew = np.expand_dims(dataset['rewards'], axis=1)
    dones = np.expand_dims(dataset['terminals'], axis=1)
    episode_step = 0
    chunk_idx = 0

    while chunk_idx < obs.shape[0] - horizon + 1:
        chunk_start_idx = chunk_idx
        chunk_end_idx = chunk_start_idx + horizon

        observations.append(torch.tensor(obs[chunk_start_idx: chunk_end_idx], dtype=torch.float32))
        actions.append(torch.tensor(act[chunk_start_idx: chunk_end_idx], dtype=torch.float32))
        if np.sum(dones[chunk_start_idx: chunk_end_idx] > 0):
            episode_step = 0
            chunk_idx += horizon
        elif (episode_step == (1000 - horizon) / stride):
            episode_step = 0
            chunk_idx += horizon
        else:
            episode_step += 1
            chunk_idx += stride

    observations = torch.stack(observations)
    actions = torch.stack(actions)

    num_samples = observations.shape[0]

    print('Total data samples extracted: ', num_samples)
    num_test_samples = int(test_split * num_samples)

    train_indices = np.arange(0, num_samples - num_test_samples)
    test_indices = np.arange(num_samples - num_test_samples, num_samples)
    np.random.shuffle(train_indices)

    observations_train = observations[train_indices]
    actions_train = actions[train_indices]

    observations_test = observations[test_indices]
    actions_test = actions[test_indices]


    return dict(observations_train=observations_train,
                actions_train=actions_train,
                # rewards_train=rewards_train,
                # terminals_train=terminals_train,
                observations_test=observations_test,
                actions_test=actions_test,
                )
