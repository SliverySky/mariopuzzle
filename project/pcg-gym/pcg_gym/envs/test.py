import gym
import pcg_gym
from pcg_gym.envs.utils import *
import numpy as np
import pcg_gym.envs.MarioLevelRepairer.GA.repairer
from pcg_gym.envs.MarioLevelRepairer.GA.repairer import Repairer
import time
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
import subprocess
pro_name = "//home//cseadmin//sty//project2//pcg-gym//pcg_gym//envs//Generator//GAN.py"
#agent = subprocess.Popen(["python", pro_name], stdout=subprocess.PIPE)
env = gym.make("mario_puzzle-v0")
env.setParameter(7)
env.reset()
v = [-1.,1.,-1.,-1.,-1.,-1.
,-1.,-1.,-1.,-1.,-1.,1.
, 1.,-1.,0.37019877,1.,1.,-1.
,-1.,1.,1.,-1.,-0.72304549,-0.62071659
, 1.,-1.,-1.,-0.32534369,-1.,-1.
,-1.,-1.        ]
cnt=0
st = time.time()
i = 0
#env.start_online_test()
#os.system("python "+pro_name)
while(True):
  #while(1):
  #  action = np.array(np.random.rand(32))
  #  playable, cost = env.online_test(action)
  #  print(i,"sample cost",cost)
  #  if playable: break
  action = np.array(np.random.rand(32))
  ob, rew, done, info = env.step(action)
  print(i,rew)
  i+=1
  if done:
    print(time.time()-st)
    env.reset()
    #saveLevelAsImage(info['recorder']["unrepair_lv"], 'un_lv',14)
    #saveLevelAsImage(info['recorder']["lv"], 'lv',14)
    #break
#env.end_online_test()
#agent.terminate()