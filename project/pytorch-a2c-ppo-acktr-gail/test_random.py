import argparse
import os
# workaround to unpickle olf model files
import sys
import gym
import numpy as np
import torch
import pcg_gym
from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from pcg_gym.envs.utils import *
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


'''if exp=='1_1':
    info={'p':14, 'q':4, 'alpha':0.2, 'object':0, 'val':0}
elif exp=='1_2':
    info={'p':14, 'q':4, 'alpha':0.2, 'object':1, 'val':0}
elif exp=='1_3':
    info={'p':14, 'q':4, 'alpha':0.2, 'object':2, 'val':0.24}
elif exp=='2_1':
    info={'p':14, 'q':4, 'alpha':0.2, 'object':2, 'val':0.24}
elif exp=='2_2':
    info={'p':28, 'q':2, 'alpha':0.2, 'object':2, 'val':0.24}
elif exp=='2_3':
    info={'p':1, 'q':56, 'alpha':0.2, 'object':2, 'val':0.24}
elif exp=='3_1':
    info={'p':14, 'q':2, 'alpha':1, 'object':2, 'val':0.45}
elif exp=='3_2':
    info={'p':14, 'q':4, 'alpha':1, 'object':2, 'val':0.53}
elif exp=='3_3':
    info={'p':14, 'q':6, 'alpha':1, 'object':2, 'val':0.57}
elif exp=='0':
    info={'p':14, 'q':6, 'alpha':1, 'object':2, 'val':0.57}'''
info = {}
info['exp'] = int(args.exp)

load_dir='./trained_models/'+'experiment'+args.exp+'/ppo/'
save_dir='./generated_levels/'+'experiment'+args.exp+"_rand/"
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
generate_num = 100
cnt = 0
env.envs[0].constrain = True
for generate_i in range(generate_num):
    masks = torch.zeros(1, 1)
    lvs = []
    obs = env.reset()
    #obs = torch.from_numpy((obs[np.newaxis, ...])).float()
    done = False
    Random_agent=True
    sum = 0
    while not done:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=False)
        if Random_agent:
            action = torch.from_numpy(np.random.rand(1,32)*2-1)
        # Obser reward and next obs
        #print('action=', action.shape, 'obs=',obs.shape)
        obs, reward, done, _ = env.step(action)
        #print('action', action)
        #print('ob', obs)
        masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32)
    win_w = env.envs[0].win_w
    #saveLevelAsText(mario.final_lv,save_dir+str(generate_i)+"_"+str(mario.rate))
    saveLevelAsText(env.envs[0].final_lv,save_dir+str(generate_i)+"_"+str(env.envs[0].rate))
    saveLevelAsImage(env.envs[0].save_lv,save_dir+"norepair_"+str(generate_i)+"_"+str(env.envs[0].rate), win_w)
    saveLevelAsImage(env.envs[0].final_lv,save_dir+str(generate_i)+"_"+str(env.envs[0].rate), win_w)
    if env.envs[0].rate==1:
        cnt += 1
    print("iter=",generate_i)
print(cnt,"/",generate_num)