import sys
from root import rootpath
sys.path.append(rootpath+'/..'+'/pytorch-a2c-ppo-acktr-gail/a2c_ppo_acktr')
sys.path.append(rootpath+'/..'+'/pytorch-a2c-ppo-acktr-gail/')

import argparse
import os
# workaround to unpickle olf model files
import gym
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
import numpy as np
import torch
import pcg_gym
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from pcg_gym.envs.utils import *
from pcg_gym.envs.latent.initial_states import *
import json
import itertools
from collections import Iterable
parser = argparse.ArgumentParser(description='RL')

parser.add_argument(
    '--env-name',
    default='mario_puzzle-v0',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--exp',
    default='1_no')
parser.add_argument(
    '--cuda',
    default='1')
args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
info = {}
info['exp'] = 4#generate a level of one hundred segments 

load_dir=rootpath + '/pretrained_agent/'+'experiment'+args.exp+'/ppo/'
save_dir=rootpath+'/levels/'+'exp'+args.exp+"/"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
env = make_vec_envs(
    args.env_name,
    1 + 1000,
    1,
    None,
    None,
    device='cuda',
    allow_early_resets=False,info=info) 
actor_critic, obs_rms = \
            torch.load(os.path.join(load_dir, args.env_name + ".pt")) #map_location='cpu'
actor_critic= actor_critic.cuda()
vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms
# intial the environment and agent

generate_num = 10
cnt = 0
recurrent_hidden_states = torch.zeros(1,
                                  actor_critic.recurrent_hidden_state_size).cuda()
Random_agent=False
for generate_i in range(generate_num):
    env.envs[0].initial_state = initial_states[generate_i] # set the initial segment
    env.envs[0].his_len=1
    for k in range(30):
        if os.path.exists(save_dir+str(generate_i)+"_"+str(k)+"_100"+".txt"):
            continue
        masks = torch.zeros(1, 1).cuda() # mask the hidden states
        lvs = []
        obs = env.reset()
        done = [False]
        sum = 0
        mark = {}
        cnt = 1
        reward_sum = 0
        while not done[0]:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=False)
            if Random_agent:
                action = torch.from_numpy(np.clip(np.random.randn(1,32), -1, 1))
            obs, reward, done, info = env.step(action)
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32).cuda()
            reward = reward.cpu().numpy()[0][0]
            cnt += 1
            reward_sum += reward
        win_w = 14
        info = info[0]
        recorder = info['recorder']
        saveLevelAsText(info['recorder']["lv"], save_dir+str(generate_i)+"_"+str(k)+"_"+str(info['ep_len']))
        #saveLevelAsImage(info['recorder']["unrepair_lv"],save_dir+"norepair_"+str(generate_i)+"_"+str(info['ep_len']), win_w)
        saveLevelAsImage(info['recorder']["lv"],save_dir+str(generate_i)+"_"+str(k)+"_"+str(info['ep_len']), win_w)
        print('sum=',reward_sum)
        print("iter=",generate_i)
