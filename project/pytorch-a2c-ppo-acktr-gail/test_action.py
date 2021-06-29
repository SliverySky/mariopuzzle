import argparse
import os
# workaround to unpickle olf model files
import sys
import gym
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
import numpy as np
import torch
import pcg_gym
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from pcg_gym.envs.utils import *
import json
import itertools
from collections import Iterable
sys.path.append('a2c_ppo_acktr')
parser = argparse.ArgumentParser(description='RL')

parser.add_argument(
    '--env-name',
    default='mario_puzzle-v0',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--exp',
    default='1')
args = parser.parse_args()

info = {}
info['exp'] = int(args.exp)
info['skip'] = 1 # skip the playability test
load_dir='./trained_models/'+'experiment'+args.exp+'/ppo/'
save_dir='./generated_levels/'+'exp'+args.exp+"/"
env = make_vec_envs(
    args.env_name,
    1 + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False,info=info)
actor_critic, obs_rms = \
            torch.load(os.path.join(load_dir, args.env_name + ".pt"),
                        map_location='cpu')
vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
generate_num = 30
cnt = 0
#env.envs[0].constrain = True
data={}
data['P_sum'] = []
data['N'] = []
data['D'] = []
data['MD'] = []
masks = torch.zeros(1, 1)
obs = env.reset()
for i in range(100):
    with torch.no_grad():
        value, action, _, _ = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=False)
    print(action, np.exp(p))