import torch
import collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def get_discount_returns(rewards, discount=1):
    returns = 0
    scale = 1
    for r in rewards:
        returns += scale * r
        scale *= discount
    return returns


def padding_episode(traj, normalizer, start, end):
    observations = normalizer(traj["observations"], "observations")
    actions = normalizer(traj["actions"], "actions")
    rewards = traj["rewards"]
    terminals = traj["terminals"]
    
    traj_len = len(rewards)
    if end > traj_len:
        observations = np.concatenate([observations, np.zeros((end - traj_len, observations.shape[1]))], 0).astype(np.float32)
        actions = np.concatenate([actions, np.zeros((end - traj_len, actions.shape[1]))], 0).astype(np.float32)
        rewards = np.concatenate([rewards, np.zeros(end - traj_len)], 0).astype(np.float32)
        terminals = np.concatenate([terminals, np.ones(end - traj_len)], 0).astype(np.float32)
    
    episode = {}
    episode["observations"] = observations[start: end]
    episode["actions"] = actions[start: end]
    episode["rewards"] = rewards[start: end]
    episode["terminals"] = terminals[start: end]
    
    return episode


class RankingBlock(object):
    def __init__(self, timestep, improve_step=10) -> None:
        self.timestep = timestep
        
        self.episodes = []
        self.episode_num = 0
        
        self.improve_step = improve_step
        self.block_indices = [[] for _ in range(self.improve_step)]
        
        self.max_len = None
        self.obs_dim = None
        self.action_dim = None
        
    def add_episode(self, episode):
        self.episodes.append(episode)
        
        if self.max_len is None:
            self.max_len = episode["observations"].shape[0]
            self.obs_dim = episode["observations"].shape[-1]
            self.action_dim = episode["actions"].shape[-1]
        
    def ranking(self):
        self.episode_returns = np.array([episode["returns"] for episode in self.episodes])
        self.ranking_indices = sorted(range(len(self.episode_returns)), key=lambda k: self.episode_returns[k])
        self.ranking_returns = self.episode_returns[self.ranking_indices]
        
        interval = (max(self.episode_returns) - min(self.episode_returns)) / self.improve_step
        
        block_idx = 0
        for idx, value in enumerate(self.ranking_returns):
            if value > min(self.episode_returns) + (block_idx + 1) * interval and block_idx < self.improve_step - 1:
                block_idx += 1
            indice = self.ranking_indices[idx]
            self.block_indices[block_idx].append(indice)
        self.block_indices = [np.array(indice) for indice in self.block_indices]
        self.block_size = np.array([len(x) for x in self.block_indices])
    
    def sample_batch(self, batch_size=10, mode="train"):
        if mode == "train":
            sample_num_per_block = max(batch_size // (self.improve_step - 1), 1)
            sample_inds = []
            for indices in self.block_indices:
                inds = np.random.choice(
                    indices,
                    size=sample_num_per_block,
                    replace=True
                )
                sample_inds.extend(list(inds))
        else:
            sample_num_per_block = batch_size
            sample_inds = np.random.choice(
                self.block_indices[-1],
                size=batch_size,
                replace=True
            )
            sample_inds = np.concatenate([sample_inds, sample_inds], axis=0)
             
        observations = np.zeros((len(sample_inds), self.max_len, self.obs_dim)).astype(np.float32)
        actions = np.zeros((len(sample_inds), self.max_len, self.action_dim)).astype(np.float32)
        for i, ind in enumerate(sample_inds):
            episode = self.episodes[ind]
            observations[i] = episode["observations"]
            actions[i] = episode["actions"]

        episodes = np.concatenate([observations, actions], axis=-1)
        
        cond_episodes = episodes[:-sample_num_per_block]
        target_episodes = episodes[sample_num_per_block:]
        init_obs = target_episodes[:, 0: 1, :self.obs_dim]
        
        return cond_episodes, target_episodes, init_obs
            
    
class SliceRankingDataset(object):
    def __init__(self, 
                 dataset, 
                 normalizer, 
                 episode_len=256, 
                 window_size=128,
                 improve_step=10, 
                 device="cuda"):
        super().__init__()
        
        self.normalizer = normalizer
        self.obs_dim = dataset["observations"].shape[-1]
        self.action_dim = dataset["actions"].shape[-1]
        self.episode_len = episode_len
        self.window_size = window_size
        self.improve_step = improve_step
        self.device = torch.device(device)

        use_timeouts = False
        if "timeouts" in dataset:
            use_timeouts = True
        
        data_ = collections.defaultdict(list)
        episode_step = 0
        self.trajs = []
        self.max_len = 0
        for i in range(dataset["rewards"].shape[0]):
            for k in ["observations", "actions", "rewards", "terminals"]:
                data_[k].append(dataset[k][i])
            episode_step += 1
            terminal = bool(dataset["terminals"][i])
            if use_timeouts:
                final_timestep = dataset["timeouts"][i]
            else:
                final_timestep = (episode_step == 1000)
            if terminal or final_timestep:
                episode_step = 0
                episode_data = {}
                for k in data_:
                    episode_data[k] = np.array(data_[k])
                self.trajs.append(episode_data)
                data_ = collections.defaultdict(list)
                self.max_len = max(self.max_len, len(episode_data["rewards"]))

        self.returns = np.array([get_discount_returns(traj["rewards"]) for traj in self.trajs])
        num_samples = np.sum([traj["rewards"].shape[0] for traj in self.trajs])
        print(f"Number of samples collected: {num_samples}")
        print(f"Num trajectories: {len(self.trajs)}")
        print(f"Trajectory returns: mean = {np.mean(self.returns)}, std = {np.std(self.returns)}, max = {np.max(self.returns)}, min = {np.min(self.returns)}")
        
        self.block_num = max((self.max_len - 1) // self.window_size, 1)
        self.blocks = [RankingBlock(self.window_size * i) for i in range(self.block_num)]
        for traj in self.trajs:
            tlen = len(traj["rewards"])
            slice_num = max((tlen - 1) // self.window_size, 1)
            for idx in range(slice_num):
                start, end = idx * self.window_size, idx * self.window_size + episode_len
                episode = padding_episode(traj, self.normalizer, start, end)
                episode["returns"] = get_discount_returns(episode["rewards"])
                self.blocks[idx].add_episode(episode)
        
        for block in self.blocks:
            block.ranking()

    def sample_batch(self, batch_size=256, mode="train"):
        sample_num_per_blocks = int(max(batch_size // self.block_num, 1))
        conds, targets, inits = [], [], []
        for block in self.blocks:
            cond, target, init = block.sample_batch(batch_size=sample_num_per_blocks, mode=mode)
            conds.append(cond)
            targets.append(target)
            inits.append(init)
        
        conds = torch.from_numpy(np.concatenate(conds, axis=0)).float().to(self.device)
        targets = torch.from_numpy(np.concatenate(targets, axis=0)).float().to(self.device)
        inits = torch.from_numpy(np.concatenate(inits, axis=0)).float().to(self.device)
        
        return conds, targets, inits
    
    def get_top_episode(self, perc):
        sample_num = int(len(self.ranking_indices) * perc)
        sample_inds = self.ranking_indices[-sample_num:]

        sample_obs = np.zeros((sample_num, self.max_len, self.obs_dim)).astype(np.float32)
        sample_act = np.zeros((sample_num, self.max_len, self.action_dim)).astype(np.float32)
        for i, ind in enumerate(sample_inds):
            episode = self.episodes[ind]
            sample_obs[i] = episode["observations"]
            sample_act[i] = episode["actions"]
        sample_obs = self.normalizer(sample_obs, "observations")
        sample_act = self.normalizer(sample_act, "actions")
        
        sample_traj = np.concatenate([sample_obs, sample_act], axis=-1)
        init_obs = sample_obs[:, 0: 1]
            
        sample_traj = torch.from_numpy(sample_traj).float().to(self.device)
        init_obs = torch.from_numpy(init_obs).float().to(self.device)
            
        return sample_traj, init_obs
    
    def sample_prefs(self, abtch_size=256):
        return  


        
    
    
    
