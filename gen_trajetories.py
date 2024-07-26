import os
import sys

sys.path.append('.')
sys.path.append('/home/ubantu/LXW/preference-diffuser')

import argparse
import pickle
import random
import numpy as np
import torch
import d4rl
import gym

from models.temporal import TrajCondUnet
from models.diffusion import GaussianDiffusion
from models.actor import DeterministicActor
from data import BlockRankingDataset
from utils.normalizer import DatasetNormalizer
from utils.logger import Logger, make_log_dirs
from utils.trainer import Trainer
from utils.render import Render
from utils.timer import Timer
from utils.evaluator import Evaluator
from utils.OTR_reward import get_otr_return
from utils.OTR_reward import OPT_rewarder_init


# load expert data
expert_file = '/home/ubantu/LXW/preference-diffuser/data/experts-halfcheetah-medium-replay-v2k-1.pkl'
with open(expert_file, 'rb') as ef:
    expert_demo = pickle.load(ef)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="bc-distance")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--domain", type=str, default="gym")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--episode-len", type=int, default=1000)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # preference
    parser.add_argument("--use-human-label", type=bool, default=False)
    parser.add_argument("--pref-episode-len", type=int, default=100)
    parser.add_argument("--pref-num", type=int, default=500)
    parser.add_argument("--dist-noise", type=float, default=0.1)
    parser.add_argument("--pref-embed-dim", type=int, default=256)

    # diffusion
    parser.add_argument("--diff-episode-len", type=int, default=1000)
    parser.add_argument("--improve-step", type=int, default=20)
    parser.add_argument("--diff-embed-dim", type=int, default=256)
    parser.add_argument("--dim-mults", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--n-diffusion-steps", type=int, default=1000)
    parser.add_argument("--upsample-temp", type=float, default=1.)
    parser.add_argument("--guidance-scale", type=float, default=1.2)

    # actor
    parser.add_argument("--actor-embed-dim", type=int, default=256)
    parser.add_argument("--actor-hidden-layer", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--actor-type", type=str, default="deterministic")
    parser.add_argument("--flow-step", type=int, default=5)
    parser.add_argument("--percentile", type=float, default=0.1)
    parser.add_argument("--actor-lr", type=float, default=3e-5)
    parser.add_argument("--threshold", type=float, default=1.02)
    parser.add_argument("--actor-max-iters", type=float, default=100)
    parser.add_argument("--actor-num-steps-per-iter", type=float, default=1000)
    parser.add_argument("--actor-batch-size", type=int, default=1024)

    return parser.parse_args()
def get_tlen(trajs, normalizer, obs_dim):
    tlens = []
    max_len = trajs.shape[1]
    trajs_unnorm = []
    for traj in trajs:
        observations = normalizer.unnormalize(traj[..., :obs_dim], "observations")
        actions = normalizer.unnormalize(traj[..., obs_dim:], "actions")
        traj_unnorm = np.hstack((observations,actions))
        trajs_unnorm.append(traj_unnorm)
        for i in range(max_len):
            if np.sum(np.abs(observations[i])) + np.sum(np.abs(actions[i])) < 1:
                tlens.append(i)
                break
            if i == max_len - 1:
                tlens.append(max_len)
    return np.array(tlens),torch.tensor(trajs_unnorm)

def is_valid(trajs, tlens, threshold=30):
    masks = np.zeros(len(trajs))
    for i, traj in enumerate(trajs):
        if np.abs(traj[:tlens[i]]).max() < threshold:
            masks[i] = 1
    return masks

def flow_to_better(diffusion_model, normalizer, trajs, init_obs, rewarder,max_flow_step=5,threshold = 1.03):
    cond = trajs
    cond_tlens,cond_unnorm = get_tlen(trajs.detach().cpu().numpy(), normalizer, diffusion_model.observation_dim)
    cond_score = get_otr_return(cond_unnorm, cond_tlens, rewarder, init_obs.shape[2])
    print("min_score: %.4f    max_score:%.4f    mean_score:%.4f" % (
    np.min(cond_score), np.max(cond_score), np.mean(cond_score)))

    generate_trajs = [cond.detach().cpu().numpy()]
    for i in range(max_flow_step):
        print("flow step:", i)
        generate_traj = diffusion_model.flow_one_step(cond, init_obs)

        tlens,gen_traj_unnorm = get_tlen(generate_traj.detach().cpu().numpy(), normalizer, diffusion_model.observation_dim)
        score = get_otr_return(gen_traj_unnorm, tlens, rewarder,init_obs.shape[2])
        valid = is_valid(generate_traj.detach().cpu().numpy(),tlens)

        indices = []
        ratio = score / cond_score
        for j in range(len(score)):
            if valid[j] and ratio[j] > threshold: 
                cond[j] = generate_traj[j]
                cond_score[j] = score[j]
                indices.append(j)

        print("improve ratio: %.4f" % (len(indices) / len(cond)))
        print("min_score: %.4f    max_score:%.4f    mean_score:%.4f" % (
        np.min(cond_score), np.max(cond_score), np.mean(cond_score)))
        generate_trajs.append(generate_traj[indices].detach().cpu().numpy())
        if (len(indices) / len(cond)) < 0.05:
            break

    return generate_trajs

def gen(args=get_args()):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    env = gym.make(args.task)
    dataset = env.get_dataset()

    datadir = f'/home/ubantu/LXW/preference-diffuser/data/halfcheetah-medium-replay-v2-otr-traj-check.pkl'
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
    dataset = OPT_BlockRankingDataset(dataset, normalizer, total_return, opt_dataset, trajs, device=args.device)

    args.obs_shape = env.observation_space.shape
    args.obs_dim = int(np.prod(args.obs_shape))
    args.action_dim = int(np.prod(env.action_space.shape))

    args.max_action = env.action_space.high[0]

    # load trained diffusion model
    temporal_model = TrajCondUnet(args.diff_episode_len, args.obs_dim + args.action_dim, hidden_dim=args.diff_embed_dim,
                                  dim_mults=args.dim_mults)
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

    diffusion_model.load_state_dict(torch.load('/home/ubantu/LXW/preference-diffuser/run_scripts/lxv/halfcheetah-medium-replay-seed4/diffusion-500.pth'))
    diffusion_model.to(args.device)
    diffusion_model.eval()

    print('-------------generate_data-------------')
    dataset.block_ranking(args.improve_step)
    
    percentile = 202 / len(dataset.trajs)
    trajs, init_obs = dataset.get_top_traj(percentile)
    flow_step = 20
    print("flow_step", flow_step)

    rewarder = OPT_rewarder_init(expert_demo,args.obs_dim)
    generate_trajs = flow_to_better(diffusion_model, normalizer, trajs, init_obs,rewarder,
                                    max_flow_step=flow_step, threshold=args.threshold)

    save_dir = '/home/ubantu/LXW/preference-diffuser/data/generate_trajs_distance/halfcheetah-medium-replay'
    for i in range(len(generate_trajs)):
        np.save(os.path.join(save_dir, 'generate_data_%d' % i), generate_trajs[i])

if __name__ == "__main__":
    gen()
