#draw diversity change curves for three type levels
import gym
import pcg_gym
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pcg_gym.envs.utils import * 
rootpath = os.path.abspath(os.path.dirname(__file__))
s=np.array([])
sx, nx = 14, 2
wd_h, wd_w = 14, 28
step = 1
color_name = ''
def draw(name):
    global s
    lv=readTextLevel(name)
    x, y = [], []
    num = (lv.shape[1]-wd_w) // step + 1
    for i in range(num):
        if i*step<sx: continue # skip segments which do not have enougth previous pieces
        x.append(i+1)
        y.append(KLWithSlideWindow(lv, (0,i*step,wd_h, wd_w), sx, nx))
    s = np.concatenate([s, y], axis=-1)
    plt.plot(x,y, color=color_name)#, color=color_name
lvs=['gan_lv', 'DCGAN/training_data/mario-1-3']#, '4-2', '1-1', '2-1', '1-3', '3-3'] #['1-1', '1-2', '1-3', '2-1', '3-1', '3-3', '4-1', '4-2', '5-1', '5-3', '6-1', '6-2', '6-3', '7-1', '8-1']
color_set=['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585']
for i in range(len(lvs)):
    color_name = color_set[i]
    draw(rootpath + '/'+lvs[i]+'.txt')
plt.legend(lvs)
plt.xlabel('Piece index (from left to right)')
plt.ylabel('Similarity beween a level piece and the generated level segment')
title='Similarity curve of level pieces of Super Mario Bros'
#plt.plot([0,180],[0.53, 0.53],'--', 'r')
plt.title(title)
plt.savefig('similarity')
print('avg=',s.mean())
print('std=',s.std())
print('cnt=',s.shape[0])