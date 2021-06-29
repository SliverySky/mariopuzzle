import json
import os
import numpy as np
from pcg_gym.envs.utils import * 
map={'X':0, 'S':1, '-':2, '?':3, 'Q':4, 'E':5,'<':6,'>':7,'[':8,']':9,'o':10,'B':11,'b':12}
result = []
win_h=7
win_w=7
for root, dirs, files in os.walk("training_data"):
    for name in files:
        path = os.path.abspath(root)+'/'+name
        lv = readTextLevel(path)
        h, w = lv.shape
        for i in range(h-win_h+1):
            for j in range(w-win_w+1):
                result.append(lv[i:i+win_h, j:j+win_w])
result = np.array(result)
print(result.shape)
with open('data_'+str(win_h)+'_'+str(win_w)+'.json','w') as f:
    json.dump(result.tolist(),f)
