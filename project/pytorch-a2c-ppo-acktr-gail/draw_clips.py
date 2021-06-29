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
info['exp'] = int(4)
load_dir='./trained_models/'+'experiment'+args.exp+'/ppo/'
save_dir='./generated_levels2/'+'clips/'
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
generate_num = 1
for generate_i in range(generate_num):
    masks = torch.zeros(1, 1)
    lvs = []
    env.envs[0].initial_state = [-0.20246206, 0.45931731,-0.52491706,-0.62602063, 0.17772184, 0.3918543
, 0.54920196,-0.69979388, 0.85800904, 0.20720996,-0.71540711, 0.08920673
,-0.599055 , -0.50612783,-0.8835962,  0.41592613,-0.97734822,-0.03503274
,-0.9653286,  0.70521526, 0.14223208,-0.15117143, 0.61507194, 0.88415996
 ,0.97553632, 0.707216,   0.63431611,-0.21817623, 0.1002697,  0.84936158
,-0.7100786, -0.19551745]
    obs = env.reset()
    print(env.envs[0].state)
    #obs = torch.from_numpy((obs[np.newaxis, ...])).float()
    done = False
    Random_agent=False
    sum = 0
    cnt=1
    while not done:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=False)
        if Random_agent:
            action = torch.from_numpy(np.random.rand(1,32)*2-1)
        # Obser reward and next obs
        #print('action=', action.shape, 'obs=',obs.shape)
        obs, reward, done, info = env.step(action)
        #print('action', action)
        #print('ob', obs)
        print(env.envs[0].cnt)
        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32)
        cnt+=1
        if cnt>=12:
            break
    win_w = 14
    info = info[0]
    #saveLevelAsText(env.envs[0].lv[0:14, 0:12*14],save_dir+args.exp)
    saveLevelAsImage(info['recorder']["unrepair_lv"],save_dir+"norepair_"+str(generate_i)+"_"+str(info['ep_len']), win_w)
    #saveLevelAsImage(env.envs[0].lv[0:14, 0:12*14],save_dir+args.exp, win_w)