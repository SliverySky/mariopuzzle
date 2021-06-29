import argparse
import os
# workaround to unpickle olf model files
import sys
import gym
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
import numpy as np
import torch
import pcg_gym
sys.path.append('/home/cseadmin/sty/project2/pytorch-a2c-ppo-acktr-gail/a2c_ppo_acktr')
sys.path.append('/home/cseadmin/sty/project2/pytorch-a2c-ppo-acktr-gail')
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from pcg_gym.envs.utils import *

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
info['visuals']=1
load_dir='/home/cseadmin/sty/project2/pytorch-a2c-ppo-acktr-gail/trained_models/'+'experiment'+args.exp+'/ppo/'
save_dir='/home/cseadmin/sty/project2/pytorch-a2c-ppo-acktr-gail/generated_levels/'+'exp'+args.exp+"/"
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
generate_num = 5
cnt = 0
env.envs[0].constrain = True
for generate_i in range(generate_num):
    masks = torch.zeros(1, 1)
    lvs = []
    obs = env.reset()
    #obs = torch.from_numpy((obs[np.newaxis, ...])).float()
    done = False
    Random_agent=False
    sum = 0
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
        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32)
    win_w = 14
    info = info[0]
    #saveLevelAsText(mario.final_lv,save_dir+str(generate_i)+"_"+str(mario.rate))
    saveLevelAsText(info['recorder']["lv"],save_dir+str(generate_i)+"_"+str(info['ep_len']))
    saveLevelAsImage(info['recorder']["unrepair_lv"],save_dir+"norepair_"+str(generate_i)+"_"+str(info['ep_len']), win_w)
    saveLevelAsImage(info['recorder']["lv"],save_dir+str(generate_i)+"_"+str(info['ep_len']), win_w)
    print("iter=",generate_i)