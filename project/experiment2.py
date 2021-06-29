import gym
import pcg_gym
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pcg_gym.envs.utils import * 
rootpath = os.path.abspath(os.path.dirname(__file__))
s=np.array([])
sx, nx = 7, 3
sy, ny = 14, 0
wd_h, wd_w = 14, 14
step = 1
color_name = ''
def draw(name):
    global s
    lv=readTextLevel(name)
    x, y = [], []
    num_y = (lv.shape[0]-wd_h) // step + 1
    num_x = (lv.shape[1]-wd_w) // step + 1
    for i in range(num_y):
        for j in range(num_x):
            if i*step<sy and j*step<sx:continue # skip segments which do not have enougth previous pieces
            x.append((i+1)*(j+1))
            y.append(KLWithSlideWindow(lv, (i*step,j*step,wd_h, wd_w), sx, nx, sy, ny))
    s = np.concatenate([s, y], axis=-1)
    #plt.plot(x,y)#, color=color_name
lvs = ['1-1', '1-2', '1-3', '2-1', '3-1', '3-3', '4-1', '4-2', '5-1', '5-3', '6-1', '6-2', '6-3', '7-1', '8-1']
#lvs = ['1-3', '3-3', '5-3' ,'6-3'] #athletic
#lvs = ['1-2', '4-2'] # underground
#lvs = ['8-1', '5-1', '6-1', '4-1', '6-2', '7-1', '2-1', '3-1', '1-1'] # overworld
#lvs = ['rand-0', 'rand-1', 'rand-2', 'rand-3', 'rand-4','rand-5','rand-6','rand-7', 'rand-8','rand-9']
color_set=['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585']
for i in range(len(lvs)):
    draw(rootpath + '/DCGAN/training_data/'+'mario-'+lvs[i]+'.txt')
#plt.legend(lvs)
#plt.xlabel('Piece index')
#plt.ylabel('KL divergence with slide windows')
#title='Diversity curve of level pieces of Super Mario Bros'
#plt.title(title)
#plt.savefig(title)
print('avg=',s.mean())
print('std=',s.std())
print('cnt=',s.shape[0])