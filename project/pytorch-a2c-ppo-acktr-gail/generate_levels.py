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
from pcg_gym.envs.latent.initial_states import *
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
save_dir='./generated_levels2/'+'exp'+args.exp+"/"
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
generate_num = 10
cnt = 0

for generate_i in range(generate_num):
    env.envs[0].initial_state = initial_states[generate_i]
    env.envs[0].his_len=1
    for k in range(10):
        masks = torch.zeros(1, 1)
        lvs = []
        obs = env.reset()
        done = False
        Random_agent=False
        sum = 0
        mark = {}
        cnt = 1
        reward_sum = 0
        while not done:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=False)
            if Random_agent:
                action = torch.from_numpy(np.random.randn(1,32)*2-1)
            # Obser reward and next obs
            #print('action=', action.shape, 'obs=',obs.shape)
            obs, reward, done, info = env.step(action)
            #print('action', action)
            #print('ob', obs)
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32)
            reward = reward.cpu().numpy()[0][0]
            if reward < 1:
                while len(info[0]['kls'])<1:
                    info[0]['kls'].append(0)
                mark[cnt]='%.2f (%.2f)' % (reward, info[0]['kls'][0])
                mark[str(cnt)+'c']=reward
            cnt += 1
            reward_sum += reward
        win_w = 14
        info = info[0]
        recorder = info['recorder']
        saveLevelAsText(info['recorder']["lv"], save_dir+str(generate_i)+"_"+str(k)+"_"+str(info['ep_len']))
        #saveLevelAsImage(info['recorder']["unrepair_lv"],save_dir+"norepair_"+str(generate_i)+"_"+str(info['ep_len']), win_w)
        saveLevelAsImage(info['recorder']["lv"],save_dir+str(generate_i)+"_"+str(k)+"_"+str(info['ep_len']), win_w, mark)
        print('sum=',reward_sum)
        print("iter=",generate_i)
