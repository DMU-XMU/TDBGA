import numpy as np
import collections
import torch
import copy
from sklearn.cluster import KMeans, MeanShift

from .human_data import Dataset
from utils.normalizer import DatasetNormalizer
from .scripted_data import ScriptedDataset

Batch = collections.namedtuple('Batch', 'observations actions masks next_observations')
ActionBatch = collections.namedtuple('ActionBatch', 'observations actions')

        
class ActorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, normalizer=None, device="cuda"):
        super().__init__()
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = DatasetNormalizer(dataset)

        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.device = torch.device(device)
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        observations = self.normalizer(self.observations[idx], 'observations')
        # actions = self.normalizer(self.actions[idx], 'actions')
        actions = self.actions[idx]
        
        return ActionBatch(observations, actions)
    
    def sample_batch(self, batch_size):
        batch_inds = np.random.choice(
            np.arange(len(self.actions)),
            size=batch_size,
            replace=True
        )
        
        observations = torch.from_numpy(self.normalizer(self.observations[batch_inds], 'observations')).float().to(self.device)
        actions = torch.from_numpy(self.actions[batch_inds]).float().to(self.device)

        return observations, actions
    

class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, normalizer=None, rnn_seq_len=10, device="cuda"):
        super().__init__()
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = DatasetNormalizer(dataset)
        
        self.rnn_seq_len = rnn_seq_len
        self.start_indices = dataset["start_indices"]
        self.end_indices = dataset["end_indices"]
        self.indices = []
        for start, end in zip(self.start_indices, self.end_indices):
            if end - start < rnn_seq_len:
                continue
            self.indices.append(np.arange(start, end-rnn_seq_len))
        self.indices = np.concatenate(self.indices)

        self.observations = dataset['observations']
        self.actions = dataset['actions']
        self.device = torch.device(device)
        
        self.num = self.observations.shape[0]
        
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        observations = self.normalizer(self.observations[idx], 'observations')
        actions = self.actions[idx]
        
        return ActionBatch(observations, actions)
    
    def sample_batch(self, batch_size):
        batch_inds = np.random.choice(
            np.arange(len(self.indices)),
            size=batch_size,
            replace=True
        )
        
        indices = np.arange(self.rnn_seq_len).reshape(1, -1).repeat(batch_size, 0) + batch_inds.reshape(-1, 1)
        
        observations = torch.from_numpy(self.normalizer(self.observations[indices], 'observations')).float().to(self.device)
        actions = torch.from_numpy(self.actions[indices]).float().to(self.device)

        return observations, actions


class BlockRankingDataset(ScriptedDataset):
    def __init__(self, dataset, normalizer, totalreturn, opt_dataset, trajs, device="cuda" ):
        super().__init__(dataset, normalizer, max_len=1000, device="cuda")
        self.returns = totalreturn
        self.opt_dataset = opt_dataset
        self.trajs = trajs

        self.observation_dim = self.obs_dim

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        observations = self.observations[idx]
        actions = self.actions[idx]
        next_observations = self.next_observations[idx]

        return Batch(observations, actions, next_observations)

    def block_ranking(self, improve_step=10):
        self.ranking_indices = sorted(range(len(self.returns)), key=lambda k: self.returns[k])
        self.ranking_returns = self.returns[self.ranking_indices]

        self.improve_step = improve_step

        km_cluster = KMeans(n_clusters=improve_step, n_init=40, init='k-means++')
        labels = km_cluster.fit_predict(self.ranking_returns.reshape(-1, 1))
        # interval = (max(self.returns) - min(self.returns)) / self.improve_step

        block_indices = [[] for _ in range(self.improve_step)]
        block_idx = 0
        for idx, value in enumerate(self.ranking_returns):
            if idx > 0 and labels[idx] != labels[idx - 1]:
                block_idx += 1
            block_indices[block_idx].append(self.ranking_indices[idx])
        self.block_indices = [np.array(indices) for indices in block_indices]
        self.block_size = np.array([len(x) for x in self.block_indices])

        for i in range(improve_step):
            indices = self.block_indices[i]
            min_return = self.returns[indices[0]]
            max_return = self.returns[indices[-1]]
            size = self.block_size[i]
            print('Block %d: min=%.4f  max=%.4f  size=%d' % (i, min_return, max_return, size))

    def get_block_id(self, score):
        for i in range(self.improve_step):
            indices = self.block_indices[i]
            min_return = self.returns[indices[0]]
            max_return = self.returns[indices[-1]]
            if min_return <= score <= max_return:
                return i
        return self.improve_step + 1

    def get_block_traj(self, idx):
        indices = self.block_indices[idx]
        observations = []
        for idx in indices:
            obs, _, _, _ = self.get_traj(idx)
            observations.append(obs)
        return np.array(observations)

    def get_top_traj(self, percentile):
        sample_num = int(len(self.trajs) * percentile)
        if sample_num > len(self.ranking_indices):
            sample_num = self.ranking_indices
        sample_inds = self.ranking_indices[-sample_num:]


        sample_obs = []
        sample_act = []
        for idx in sample_inds:
            observations, actions, _, _ = self.get_traj(idx)
            sample_obs.append(observations)
            sample_act.append(actions)
        sample_obs = np.array(sample_obs)
        sample_act = np.array(sample_act)
        sample_traj = np.concatenate([sample_obs, sample_act], axis=-1)

        condition_traj = torch.from_numpy(sample_traj).float().to(self.device)
        init_obs = condition_traj[:, 0: 1, :self.observation_dim]

        return condition_traj, init_obs

    def sample_batch(self, batch_size, mode="train"):
        if mode == "train":
            sample_size_per_block = np.ceil(batch_size / (self.improve_step - 1)).astype(int)

            sample_inds = []
            for indices in self.block_indices:
                batch_inds = np.random.choice(
                    np.arange(len(indices)),
                    size=sample_size_per_block,
                    replace=True
                )
                sample_inds.extend(indices[batch_inds])
            sample_inds = sample_inds
        else:
            batch_inds = np.random.choice(
                np.arange(len(self.block_indices[-1])),
                size=batch_size,
                replace=True
            )
            sample_inds = self.block_indices[-1][batch_inds]

        sample_obs = []
        sample_act = []
        sample_mask = []
        for idx in sample_inds:
            observations, actions, _, masks = self.get_traj(idx)
            sample_obs.append(observations)
            sample_act.append(actions)
            sample_mask.append(masks)
        sample_obs = np.array(sample_obs)
        sample_act = np.array(sample_act)
        sample_mask = np.array(sample_mask)
        sample_traj = np.concatenate([sample_obs, sample_act], axis=-1)

        if mode == "train":
            condition_traj = torch.from_numpy(sample_traj[:-sample_size_per_block]).float().to(self.device)
            target_traj = torch.from_numpy(sample_traj[sample_size_per_block:]).float().to(self.device)
            masks = torch.from_numpy(sample_mask[sample_size_per_block:]).float().to(self.device).unsqueeze(dim=-1)
        else:
            condition_traj = torch.from_numpy(sample_traj).float().to(self.device)
            target_traj = torch.from_numpy(sample_traj).float().to(self.device)
            masks = torch.from_numpy(sample_mask).float().to(self.device).unsqueeze(dim=-1)
        init_obs = target_traj[:, 0: 1, :self.observation_dim]

        return condition_traj, target_traj, init_obs, masks
