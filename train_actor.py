import pickle
import os
import sys
sys.path.append('.')
sys.path.append('/home/ubantu/LXW/preference-diffuser')
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="OPT-Diffusion")
    parser.add_argument("--domain", type=str, default="d4rl")
    parser.add_argument("--task", type=str, default="antmaze-umaze-v0")
    parser.add_argument("--data-type", type=str, default="low")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-len", type=int, default=1000)
    parser.add_argument("--discount", type=float, default=1)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")

    # actor
    parser.add_argument("--actor-embed-dim", type=int, default=256)
    parser.add_argument("--actor-hidden-layer", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--actor-type", type=str, default="deterministic")
    parser.add_argument("--select-num", type=int, default=300)   
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=1.05)
    parser.add_argument("--actor-max-iters", type=float, default=100)
    parser.add_argument("--actor-num-steps-per-iter", type=float, default=1000)
    parser.add_argument("--actor-batch-size", type=int, default=256) 

    return parser.parse_args()

def train(args=get_args()):
    experiment = Experiment(api_key='ZT2J72DisEsEgG1CPtTK6T5A8',
                        project_name='preference-trainactor')
    # set seed everywhere
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

    actor = DeterministicActor(
            observation_dim=args.obs_dim,
            action_dim=args.action_dim,
            hidden_dim=args.actor_embed_dim,
            hidden_layer=args.actor_hidden_layer
        )
    actor.to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr, weight_decay=args.weight_decay)
    actor.train()

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

    timer = Timer()
    evaluator = Evaluator(env, normalizer)

    with open('/home/ubantu/LXW/preference-diffuser/data/bc_dataset/antmaze-umaze-v0-generate2BC-episode1000.pkl','rb') as file:
            bc_dataset = pickle.load(file)

    
    bc_dataset = ActorDataset(bc_dataset, normalizer, device=args.device)

    trainer = BCTrainer(actor,bc_dataset,logger,timer,evaluator,experiment,args.device)
    print('-------------train actor-------------')
    trainer.train(
        optim=actor_optim,
        max_iters=args.actor_max_iters,
        num_steps_per_iter=args.actor_num_steps_per_iter,
        batch_size=args.actor_batch_size,
    )

if __name__ == "__main__":
    train()