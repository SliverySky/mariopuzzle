import argparse
import os
# workaround to unpickle olf model files
import sys
import gym
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
from pcg_gym.envs.search_latent import *
from pcg_gym.envs.utils import *
from pcg_gym.envs.generator2 import *
import os
root = os.path.abspath(os.path.dirname(__file__))
#model_path = root + "//models//14_14.pth"
generator = Generator(0)
env = gym.make("mario_puzzle-v0")
env.setParameter({'exp':11})
for i in range(20,30):
    obs = env.reset()
    lv = generator.generate(obs)
    saveLevelAsImage(lv, root+"//latent//"+str(i))
    s = "[%.10f"%obs[0]
    for i in range(1,obs.shape[0]):
        s += ",%.10f"%obs[i]
    s+="]"
    print(s,",")
