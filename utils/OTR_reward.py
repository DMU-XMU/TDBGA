import numpy as np
import tqdm
import functools
from .otil import rewarder as rewarder_lib
from typing import Any, NamedTuple

NestedArray = Any


class tempTransition(NamedTuple):
    observation: NestedArray
    action: NestedArray


def relabel_rewards(rewarder, trajectory, tlen, obs_dim):

    otr_trajs = []
    obsdim = obs_dim
    for i in range(tlen):
        otr_trajs.append(
            tempTransition(
                observation=trajectory[i][:obsdim],
                action=trajectory[i][obsdim:]
            )
        )

    rewards = rewarder.compute_offline_rewards(otr_trajs)
    return rewards.sum()


def OPT_rewarder_init(expert_demo, obs_dim):
    episode_length = 1000
    atom_size = obs_dim
    squashing_fn = functools.partial(
        rewarder_lib.squashing_exponential,
        alpha=5.0,  # 5.0
        beta=5.0 * episode_length / atom_size)

    rewarder = rewarder_lib.OTILRewarder(
        expert_demo, episode_length=episode_length, squashing_fn=squashing_fn)
    return rewarder


def get_otr_return(generate_trajs, tlens, rewarder, obs_dim):
    final_return = []
    for i in tqdm.trange(len(generate_trajs)):  # pylint: disable=all
        final_return.append(relabel_rewards(rewarder, generate_trajs[i], tlens[i], obs_dim))
    return np.array(final_return)


def total_return(relabeled_trajectories):
    total_return = []
    trajs = []
    for traj in relabeled_trajectories:
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

