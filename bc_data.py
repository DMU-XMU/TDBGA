import os
import sys
from comet_ml import Experiment
import argparse
import pickle
import random
import numpy as np
import torch
import gym
import d4rl
from models.actor import DeterministicActor
from utils.normalizer import DatasetNormalizer
from data import ActorDataset
from utils.timer import Timer
from utils.logger import Logger, make_log_dirs
from utils.trainer import BCTrainer
from utils.evaluator import Evaluator
from utils.OTR_reward import get_otr_return, OPT_rewarder_init

def get_tlen(trajs, normalizer, obs_dim):

    tlens = []
    max_len = trajs.shape[1]
    trajs_unnorm = []
    for traj in trajs:
        observations = normalizer.unnormalize(traj[..., :obs_dim], "observations")
        actions = normalizer.unnormalize(traj[..., obs_dim:], "actions")
        traj_unnorm = np.hstack((observations, actions))
        trajs_unnorm.append(traj_unnorm)
        for i in range(max_len):
            if np.sum(np.abs(observations[i])) + np.sum(np.abs(actions[i])) < 1:
                tlens.append(i)
                break
            if i == max_len - 1:
                tlens.append(max_len)
    return np.array(tlens), torch.tensor(trajs_unnorm)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="OPT-Diffusion")
    parser.add_argument("--domain", type=str, default="d4rl")
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--data-type", type=str, default="low")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-len", type=int, default=1000)
    parser.add_argument("--discount", type=float, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--select-num", type=int, default=300)   


    return parser.parse_args()

def generate(args=get_args()):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    env = gym.make(args.task)
    dataset = env.get_dataset()
    normalizer = DatasetNormalizer(dataset)  
    args.obs_shape = env.observation_space.shape
    args.obs_dim = int(np.prod(args.obs_shape))
    args.action_dim = int(np.prod(env.action_space.shape))
    args.max_action = env.action_space.high[0]
    
    expert_file = '/home/ubantu/LXW/preference-diffuser/data/experts-halfcheetah-medium-replay-v2k-1.pkl'
    with open(expert_file, 'rb') as ef:
        expert_demo = pickle.load(ef)

    gen_num = 1 
    generate_trajs = []
    for i in range(gen_num):
        temp_data = np.load(f'/home/ubantu/LXW/preference-diffuser/data/generate_trajs_distance/halfcheetah-medium-replay/generate_data_{i}.npy')
        generate_trajs.append(temp_data)
    generate_trajs = np.concatenate(generate_trajs)
    print(' generate trajs num is {}',len(generate_trajs))

    rewarder = OPT_rewarder_init(expert_demo, args.obs_dim)
    tlens, generate_trajs_unnorm = get_tlen(generate_trajs, normalizer, args.obs_dim)
    bc_trajs = []
    for i in range(len(tlens)):
        bc_trajs.append(generate_trajs[i][:tlens[i]])
    bc_trajs = np.concatenate(bc_trajs)
    bc_dataset = dict()
    bc_dataset["observations"] = normalizer.unnormalize(bc_trajs[..., :args.obs_dim], "observations").reshape(-1, args.obs_dim)
    bc_dataset["actions"] = normalizer.unnormalize(bc_trajs[..., args.obs_dim:], "actions").reshape(-1, args.action_dim)
    with open('../data/bc_dataset/' + args.task + '-generate2BC' + '.pkl','wb') as file:
        pickle.dump(bc_dataset,file)

if __name__ == '__main__':
    generate()