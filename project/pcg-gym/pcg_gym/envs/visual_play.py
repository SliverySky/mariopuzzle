import pcg_gym, gym
from pcg_gym.envs.utils import *
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
import numpy as np
env = gym.make("mario_puzzle-v0")
info = {"exp":3, "visuals":1} # use playability reward only
env.setParameter(info)
env.reset()
for i in range(10000):
    ob, rew, done ,info = env.step(np.random.rand(32))
    if done:
        env.reset()
        print("Game end. The sum of playablity reward equals", rew)
#lv = readTextLevel("/home/cseadmin/sty/project2/Experiment/levels/exp7_online/21.txt")#readTextLevel("/home/cseadmin/sty/project2/Experiment/levels/exp1/5_0_100.txt")
#p = env.agent.start_test(lv)
#print(p)
#for i in range(0, lv.shape[1]//14-1):
#    now_x = i*14+14
#    p=env.agent.continue_test(lv[:, max(0, now_x-3*14): now_x+14])
#    print(p)


