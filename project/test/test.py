import pcg_gym, gym
from pcg_gym.envs.utils import *
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
import numpy as np
env = gym.make("mario_puzzle-v0")
import json
info = {"exp":7}
#info = {"exp":14}
env.setParameter(info)
env.reset()
data = {}
def add(name, elm):
    if name not in data.keys():
        data[name]=[]
    else: data[name].append(elm)
for i in range(100000):
    ob, rew, done ,info = env.step(np.clip(np.random.randn(32), -1, 1))
    if done:
        add('x', i)
        add('r_norm', info["rewD_sum"]+info["rewN_sum"]+info["rewP_sum"])
        add('r', info["rewP_sum"]+info["MD_sum"]+info["N_sum"])
        add('N_max', info['N_max'])
        add('N_min', info['N_min'])
        add('MD_max', info['MD_max'])
        add('MD_min', info['MD_min'])
        with open("data.json","w") as f:
            json.dump(data,f)
        print(i)
        env.reset()