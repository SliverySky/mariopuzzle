import pcg_gym.envs.models.dcgan as dcgan
import torch
from torch.autograd import Variable
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
rootpath = os.path.abspath(os.path.dirname(__file__))
import json
import time
class Generator():
    def __init__(self, id):
        self.id = id
        self.cnt = 0
    def generate(self, noise):
        path = rootpath+"//Generator//input//"+str(self.id)+"_"+str(self.cnt)
        with open(path,"w") as f:
            json.dump(noise.tolist(),f)
        while True:
            name = rootpath+"//Generator//output//"+str(self.id)+"_"+str(self.cnt)
            if(os.path.exists(name) and os.path.getsize(name)>0):
                with open(name) as f:
                    data = json.load(f)
                os.remove(name)
                break
            time.sleep(0.01)
        self.cnt += 1
        return np.array(data)