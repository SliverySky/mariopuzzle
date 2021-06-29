import os, sys
import time
import gym
from gym import spaces
import pcg_gym.envs.models.dcgan as dcgan
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
import torch, json, numpy
import random
import numpy as np
import math
import os
from collections import deque
import subprocess, sys, copy
from pcg_gym.envs.generator2 import Generator
from pcg_gym.envs.utils import *
from pcg_gym.envs.MarioLevelRepairer.GA.repairer import Repairer
rootpath = os.path.abspath(os.path.dirname(__file__))
from atexit import register
from subprocess import *
import subprocess
import time
import socket
gpu_num = 4 # each environment contains a GAN and a repairer. They are distributed into multiple gpus.

def clean(prog):
    try:
        prog.close()
    except:
        pass
class AStarAgent():
    # @atexit.register
    def __init__(self, id, visuals):
        self.id = id
        self.test_id = 0
        self.visuals = '1' if visuals else '0'
    def start(self):
        #self.agent = subprocess.Popen(["java", "-jar",rootpath + "/Mario-AI-Framework-master.jar", rootpath + "/" + str(self.id), str(port)], stdin=PIPE, stdout=PIPE)
        register(clean, self)
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # find a unused port 
        while True:
            self.port = int(random.random()*5000)+2000
            try:
                self.client.connect(('localhost', self.port))
                self.client.shutdown(2)
            except:
                break
        
        os.system("java -jar "+rootpath + "/Mario-AI-Framework-master.jar " + rootpath + "/test_pool/" + str(self.id) + " " +str(self.port)+" "+self.visuals+" &")
        while True:
            try:
                self.client.connect(('localhost',self.port))
                break
            except:
                pass
    def close(self):
        msg = 'close' 
        self.client.send(msg.encode('utf-8'))   
        self.client.close() 
        self.test_id = 0
    def start_test(self, lv):
        return self.__test(lv, "start\n")
    def continue_test(self, lv):
        return self.__test(lv, "continue\n")
    def retest(self, lv):
        return self.__test(lv, "retest\n")
    def __test(self, lv, msg):
        name = str(self.id)+"_"+str(self.test_id)
        saveLevelAsText(lv, rootpath+"/test_pool/"+name)
        res = self.client.send(msg.encode('utf-8'))  
        data = self.client.recv(1024) 
        rate = float(data.decode())
        os.remove(rootpath+"/test_pool/"+name+".txt")
        self.test_id+=1
        return rate
class MarioPuzzle(gym.Env):
    def __init__(self):
        self.id = random.randint(1,10000000)
        self.nz = 32
        self.elm_num = 13
        self.action_space = spaces.Box(-1., 1., shape=(self.nz,), dtype='float32')
        self.observation_space = spaces.Box(-1., 1., shape=(self.nz,), dtype='float32')
        self.D_que = deque(maxlen=1000)
        self.N_que = deque(maxlen=1000)
        self.use_P, self.use_D, self.use_N = False, False, False
        self.pop = deque(maxlen=20)
        self.novel_k = 10
        self.initial_state = None
        #recoard information
        self.recorder = {}
        self.start_time = 0
        self.online = False
        self.mutation = False
        self.skip = False

    def kl_fn(self, val):
        if (val<0.26):return -(val-0.26)**2
        if (val>0.94):return -(val-0.94)**2
        return 0

    def setParameter(self, info, index=0):
        exp = info['exp']
        visuals = 'visuals' in info.keys()
        self.skip = 'skip' in info.keys()
        # set reward function
        self.cnt = 0
        self.map_h = 14
        self.map_w = 14*100
        self.win_h, self.win_w = 14, 14
        self.sy, self.sx = 14, 7
        self.ny, self.nx = 0, 3
        if exp==1: # D
            self.use_D, self.use_N, self.use_P=True, False, False
        elif exp==2:# N
            self.use_D, self.use_N, self.use_P=False, True, False
        elif exp==3:# P
            self.use_D, self.use_N, self.use_P=False, False, True
        elif exp==4:# DN
            self.use_D, self.use_N, self.use_P=True, True, False
        elif exp==5:# DP
            self.use_D, self.use_N, self.use_P=True, False, True
        elif exp==6:# NP
            self.use_D, self.use_N, self.use_P=False, True, True
        elif exp==7:# DNP
            self.use_D, self.use_N, self.use_P=True, True, True
        self.norm = (exp>3)
        self.agent = AStarAgent(self.id, visuals)
        self.agent.start()
        # other settings
        model_path = os.path.dirname(__file__) + "//models//" + str(self.win_h)+"_"+str(self.win_w)+".pth"
        self.generator = Generator(self.id, index//gpu_num)
        self.repairer = Repairer(index//gpu_num)
        self.tot = (self.map_h // self.win_h) * (self.map_w // self.win_w)
    def sampleRandomVector(self, size):
        return np.random.rand(size)*2-1
    def add_then_norm(self, value, history):
        if not self.norm:return value
        history.append(value)
        maxv = max(history)
        minv = min(history)
        if maxv == minv:
            return 0
        else:
            return (value-minv)/(maxv-minv)

    def reset(self):
        recorder = self.recorder = {}
        recorder['defect'] = [] # the number of defects for each segment
        recorder['sample'] = [] # the resample time for each segment
        recorder['time'] = [] # the time needed to generate each segment
        recorder['D'] = [] # diversity
        recorder['N'] = [] # novelty
        recorder['MD'] = []
        recorder['rew_MD'] = [] # normalized
        recorder['rew_N'] = []
        recorder['rew_P'] = []
        self.start_time = time.time()
        self.lv = np.zeros(shape=(self.map_h, self.map_w), dtype=np.uint8)
        self.unrepair_lv = np.zeros(shape=(self.map_h, self.map_w), dtype=np.uint8)
        playable = False
        # run the A* agent to test segments
        while not playable:
            if self.initial_state != None:
                self.state = self.initial_state
            else:
                self.state = self.sampleRandomVector(self.nz)
            st = time.time()
            piece = self.generator.generate(self.state) # generate the segment
            st = time.time()
            self.unrepair_lv[0:self.win_h, 0:self.win_w] = piece
            new_piece = self.repairer.repair(piece) # repair the segment
            rate = self.agent.start_test(new_piece) # test the segment
            if rate ==1.0:
                recorder['sample'].append(1)
                playable=True
            else:recorder['sample'].append(0)
            recorder['time'].append(time.time()-self.start_time)
            self.start_time=time.time()
        self.lv[0:self.win_h, 0:self.win_w] = new_piece # fill the segment
        self.cnt = 1
        self.last_play = True
        self.pop.clear()
        self.pop.append(lv2Map(self.lv[0:self.win_h, 0:self.win_w]))
        return self.state
    def step(self, action):
        if self.online: return self.online_test(action)
        recorder = self.recorder
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state = action
        now_x = (self.cnt // (self.map_h // self.win_h))* self.win_w
        piece = self.generator.generate(self.state)
        self.lv[:, now_x : now_x+self.win_w] = piece
        self.unrepair_lv[:, now_x : now_x+self.win_w] = piece
        # repair two adjacent shor segments.
        self.lv[:, now_x-self.win_w : now_x+self.win_w]= self.repairer.repair(self.lv[:, now_x-self.win_w : now_x+self.win_w])
        reward, done = 0, False
        if self.use_P:
            rate = self.agent.continue_test(self.lv[:, max(0, now_x-3*self.win_w): now_x+self.win_w])
            reward += rate
            if rate<1.0: done=True
            recorder['rew_P'].append(rate)
        kl_val = KLWithSlideWindow(self.lv, (0, now_x, self.win_h, self.win_w), self.sx, self.nx, self.sy, self.ny)
        if self.use_D:
            recorder['D'].append(kl_val)
            recorder['MD'].append(self.kl_fn(kl_val))
            rew_D = self.add_then_norm(self.kl_fn(kl_val), self.D_que)
            recorder['rew_MD'].append(rew_D)
            reward += rew_D
        if self.use_N:
            piece_mp = lv2Map(self.lv[:, now_x: now_x + self.win_w])
            novelty = self.cal_novelty(piece_mp)
            recorder['N'].append(novelty)
            rew_N = self.add_then_norm(novelty, self.N_que)
            recorder['rew_N'].append(rew_N)
            reward += rew_N
            self.pop.append(piece_mp)
        self.cnt += 1
        if self.cnt >= self.tot:done=True
        info = {}
        if done:
            info['rewD_sum'] = sum(recorder['rew_MD'])
            info['rewN_sum'] = sum(recorder['rew_N'])
            info['rewP_sum'] = sum(recorder['rew_P'])
            info['MD_sum'] = sum(recorder['MD'])
            info['N_sum'] = sum(recorder['N'])
            info['ep_len'] = self.cnt
            #info['N_max'] =  max(self.N_que)
            #info['N_min'] = min(self.N_que)
            #info['MD_max'] = max(self.D_que)
            #info['MD_min'] = min(self.D_que)
            self.recorder['lv'] = self.lv[:,0:now_x+self.win_w]
            self.recorder['unrepair_lv']=self.unrepair_lv[:,0:now_x+self.win_w]
            info['recorder']=recorder
        # calculate novelty socre for this piece
        return self.state, reward, done, info
    # A* agent is used for online test (may be surrgate model in the future)
    def online_test(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        recorder = self.recorder
        self.start_time = time.time()
        info = {}
        now_x = (self.cnt // (self.map_h // self.win_h))* self.win_w
        if not self.last_play:
            self.lv = self.last_lv
        self.last_lv = copy.deepcopy(self.lv)
        piece = self.generator.generate(action)
        self.lv[:, now_x : now_x+self.win_w] = piece
        self.unrepair_lv[:, now_x : now_x+self.win_w] = piece
        # repair two adjacent shor segments.
        self.lv[:, now_x-self.win_w : now_x+self.win_w]= self.repairer.repair(self.lv[:, now_x-self.win_w : now_x+self.win_w])
        
        if self.last_play: rate = self.agent.continue_test(self.lv[:, max(0, now_x-3*self.win_w):now_x+self.win_w])
        else: rate = self.agent.retest(self.lv[:, max(0, now_x-3*self.win_w):now_x+self.win_w])
        if rate==1:
            self.cnt+=1
            self.last_play=True
            recorder['sample'].append(1)
            self.state = action
        else:
            self.last_play=False
            recorder['sample'].append(0)
        recorder['time'].append(time.time()-self.start_time)
        done = False
        info['playable']=self.last_play
        resample_cnt = 0
        for i in range(len(recorder['sample'])):
            if recorder['sample'][len(recorder['sample'])-1-i] == 0:
                resample_cnt += 1
            else: break
        
        if self.cnt >= self.tot or resample_cnt>=20:
            recorder['defect_before'] = self.repairer.cal_draw_defect(self.unrepair_lv)
            recorder['defect_end'] = self.repairer.cal_draw_defect(self.lv)
            info['recorder'] = recorder
            info['lv'] = self.lv
            info['unrepair_lv'] = self.unrepair_lv
            done=True
        return self.state, 0, done, info

    def cal_novelty(self, piece):
        score=[]
        for x in self.pop:
            score.append(calKLFromMap(x, piece))
        score.sort()
        sum = 0
        siz = min(len(score), self.novel_k)
        for i in range(siz):
            sum += score[i]
        if siz>0: sum /= siz
        return sum