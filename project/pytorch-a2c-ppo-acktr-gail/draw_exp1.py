import gym
import pcg_gym
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def read_lv(name):
    map={'X':0, 'S':1, '-':2, '?':3, 'Q':4, 'E':5,'<':6,'>':7,'[':8,']':9,'o':10,'B':11,'b':12}
    result = []
    with open(name,'r') as f:
        data = f.readlines()
        h, w = len(data), len(data[0])-1
        arr = np.empty(shape=(h,w), dtype=int)
        for i in range(h):
            for j in range(w):
                arr[i][j]=map[data[i][j]]
    return arr
def draw(exp, color1='', color2=''):
    env = gym.make("mario_gan-v0")
    if exp=='1_1':
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
    env.set_klsw(info['p'], info['q'], info['alpha'])
    t=10
    data = np.zeros(shape=(t, 30), dtype=np.float32)
    
    for k in range(30):
        path = "./generated_levels/exp"+exp+"/"+str(k)+".txt"
        lv = read_lv(path)
        for i in range(t):
            data[i][k]=env.lv_klsw(lv[0:14,0:(i+1)*28])
    mean = np.mean(data, axis=-1)
    std = np.std(data, axis=-1)
    if color1!='':
        plt.plot([i for i in range(t)], mean, color=color1)
    else:
        plt.plot([i for i in range(t)], mean, )
    if color2!='':
        plt.fill_between([i for i in range(t)], mean-std, mean+std, facecolor=color2)
    else:
        plt.fill_between([i for i in range(t)], mean-std, mean+std, facecolor='#D3D3D3')


exps=["2_1","2_2","2_3"]
#for exp in exps:
#    draw(exp)
draw("2_1", "#008000", "#90EE90")
draw("2_2", '#DC143C','#FFB6C1')
draw("2_3", '#0000FF','#6495ED')
plt.legend(exps)
plt.xlabel('segment index')
plt.ylabel('klsw')
title= "Experiment2"
plt.title(title)
plt.savefig(title)