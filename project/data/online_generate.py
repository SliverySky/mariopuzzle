import sys
sys.path.append('/home/cseadmin/sty/project2/pytorch-a2c-ppo-acktr-gail/a2c_ppo_acktr')
sys.path.append('/home/cseadmin/sty/project2/pytorch-a2c-ppo-acktr-gail/')
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
from pcg_gym.envs.latent.initial_states import *

sys.path.append('a2c_ppo_acktr')
parser = argparse.ArgumentParser(description='RL')

parser.add_argument(
    '--env-name',
    default='mario_puzzle-v0',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--exp',
    default='7')
parser.add_argument(
    '--random',
    default=0
)
args = parser.parse_args()

info = {}
info['exp'] = int(7)
random_resample = True if int(args.random)==1 else False
load_dir='/home/cseadmin/sty/project2/pytorch-a2c-ppo-acktr-gail/trained_models/'+'experiment'+args.exp+'/ppo/'
save_dir='./levels/'+'exp'+args.exp+"_online"+("_random"if random_resample else "")+"/"
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
generate_i = 0
env.envs[0].online = True
for index in range(generate_num):
    env.envs[0].initial_state = initial_states[index] # set the initial segment
    env.envs[0].his_len=1
    for k in range(10):
        if os.path.exists(save_dir+str(generate_i)+".txt"):
            generate_i += 1
            continue
        masks = torch.zeros(1, 1)
        lvs = []
        obs = env.reset()
        #obs = torch.from_numpy((obs[np.newaxis, ...])).float()
        done = [False]
        Random_agent=False
        sum = 0
        cnt = 0
        while not done[0]:
            with torch.no_grad():
                value, action, _, new_recurrent_hidden_states = actor_critic.act(
                    obs, recurrent_hidden_states, masks, deterministic=False)
            if cnt>0 and not info[0]['playable'] and random_resample:
                action = torch.from_numpy(np.clip(np.random.randn(1,32), -1, 1))
            if cnt>0 and info[0]['playable']: #only when the segment is playable, the hidden state is updated.
                recurrent_hidden_states = new_recurrent_hidden_states 
            obs, reward, done, info = env.step(action)
            masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32)
            cnt+=1
        info = info[0]
        #print(info)
        recorder = info['recorder']
        defect_before = recorder['defect_before']
        defect_end = recorder['defect_end']
        lv_len = info["unrepair_lv"].shape[0]//14
        saveLevelAsText(info["unrepair_lv"],save_dir+ str(generate_i)+"_unrepair" )
        #saveLevelAsImage(info["unrepair_lv"],save_dir+ str(generate_i)+"_unrepair", 14)
        #saveAndMark(info["unrepair_lv"],save_dir+ str(generate_i)+"_unrepair_mark", defect_before[0], defect_before[1])
        saveLevelAsText(info["lv"],save_dir+ str(generate_i) )
        saveLevelAsImage(info["lv"],save_dir+ str(generate_i)+"_"+str(lv_len),14)
        #saveAndMark(info["lv"],save_dir+ str(generate_i)+"_mark", defect_end[0], defect_end[1])
        info["repair_tile"] = int(np.sum(np.where(info["lv"]==info["unrepair_lv"],0,1)))
        info["unrepair_lv"] = []
        info["lv"] = []
        info["terminal_observation"] = []
        
        for x in info.keys():
            print(x, type(info[x]))
        with open(save_dir+str(generate_i)+"_info", "w") as f:
            json.dump(info, f)
        #saveLevelAsText(mario.final_lv,save_dir+str(generate_i)+"_"+str(mario.rate))
        #saveAndMark()
        #saveLevelAsText(env.envs[0].save["lv"],save_dir+str(generate_i)+"_"+str(rate))
        #saveLevelAsImage(env.envs[0].save["unrepair_lv"],save_dir+"norepair_"+str(generate_i)+"_"+str(rate), win_w)
        #saveLevelAsImage(env.envs[0].save["lv"],save_dir+str(generate_i)+"_"+str(rate), win_w)
        print("iter=",generate_i)
        generate_i += 1
print(cnt,"/",generate_num)