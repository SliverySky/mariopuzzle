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
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #声明socket类型，同时生成链接对象
        # find a unused port 
        while True:
            self.port = int(random.random()*10000)+2000
            try:
                self.client.connect(('localhost', self.port))
                self.client.shutdown(2)
            except:
                break
        
        os.system("java -jar "+rootpath + "/Mario-AI-Framework-master.jar " + rootpath + "/test_pool/" + str(self.id) + " " +str(self.port)+" "+self.visuals+" &")
        while True:
            try:
                self.client.connect(('localhost',self.port)) #建立一个链接，连接到本地的6969端口
                break
            except:
                pass
    def close(self):
        msg = 'close'  #strip默认取出字符串的头尾空格
        self.client.send(msg.encode('utf-8'))  #发送一条信息 python3 只接收btye流  
        self.client.close() #关闭这个链接
        self.test_id = 0
    def start_test(self, lv):
        return self.__test(lv, "start\n")
    def continue_test(self, lv):
        return self.__test(lv, "continue\n")
    def retest(self, lv):
        return self.__test(lv, "retest\n")
    def __test(self, lv, msg):
            # addr = client.accept()
            # print '连接地址：', addr
        name = str(self.id)+"_"+str(self.test_id)
        saveLevelAsText(lv, rootpath+"/test_pool/"+name)
        res = self.client.send(msg.encode('utf-8'))  #发送一条信息 python3 只接收btye流
        #阻塞等待
        data = self.client.recv(1024) #接收一个信息，并指定接收的大小 为1024字节
        #print(data.decode())
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
        self.exp = 0
        self.his_len = 1
    def kl_fn(self, val, i):
        lb = np.array([0.47, 0.48, 0.50, 0.49, 0.55, 0.56])
        ub = np.array([1.29, 1.47, 1.46, 1.41, 1.71, 1.65])
        maxv = np.array([6.54, 6.29, 6.43, 5.71, 6.49, 6.33])
        alpha = lb
        beta = maxv - ub
        if (val<lb[i]):return 1-((lb[i]-val))**0.5
        if (val>ub[i]):return 1-((val-ub[i]))**0.5
        return 1

    def setParameter(self, info, index=0):
        exp = info['exp']
        self.exp = exp
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
        elif exp==11: # 
            self.his_len = 1
        elif exp==12:
            self.his_len = 2
        elif exp==13:
            self.his_len = 5
        elif exp==14:
            self.his_len = 1
            self.map_w = 50 * 14
        elif exp==15:
            self.his_len = 1
            self.map_w = 50 * 14
            self.use_P = True
        elif exp==16:
            self.his_len = 5
            self.map_w = 50*14
        elif exp==17:
            self.his_len = 2
            self.map_w = 50*14
        elif exp==18: # max diversity
            self.skip = 1
        self.agent = AStarAgent(self.id, visuals)
        self.agent.start()
        # other settings
        model_path = os.path.dirname(__file__) + "//models//" + str(self.win_h)+"_"+str(self.win_w)+".pth"
        #self.generator = Generator(model_path, self.win_h, self.win_w, index//8)
        self.generator = Generator(self.id, index//4)
        self.repairer = Repairer(index//4)
        self.tot = (self.map_h // self.win_h) * (self.map_w // self.win_w)
    def sampleRandomVector(self, size):
        return np.clip(np.random.randn(size), self.action_space.low, self.action_space.high)
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
        self.mp = []
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
            if rate == 1.0:
                recorder['sample'].append(1)
                playable=True
            else:recorder['sample'].append(0)
            recorder['time'].append(time.time()-self.start_time)
            self.start_time=time.time()
        self.lv[0:self.win_h, 0:self.win_w] = new_piece # fill the segment
        self.mp.append(lv2Map(new_piece))
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
        fun = 0
        num = min(self.his_len, self.cnt)
        kls = []
        fac, decay = 1, 0.9
        s = 0
        for i in range(1, num+1):
            val = calKLFromMap(lv2Map(self.lv[:, now_x: now_x+self.win_w]), lv2Map(self.lv[:, now_x-i*self.win_w: now_x+(1-i)*self.win_w]))
            kls.append(val)
            reward += fac * self.kl_fn(val, i-1)
            s += fac
            fac *= decay
        reward /= s
        if self.skip:
            rate = 1.0
        else:
            rate = self.agent.continue_test(self.lv[:, max(0, now_x-3*self.win_w): now_x+self.win_w])
        recorder['rew_P'].append(rate)
        recorder['rew_MD'].append(reward)
        if self.use_P:
            reward = 1
        if rate<1.0: 
            done = True
            reward = 0
        if self.exp == 18:
            reward = KLWithSlideWindow(self.lv, (0, now_x, self.win_h, self.win_w), self.sx, self.nx, self.sy, self.ny)
        '''if self.use_D:
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
            self.pop.append(piece_mp)'''
        self.cnt += 1
        if self.cnt >= self.tot: done=True
        info = {}
        info['kls'] = kls
        if done:
            info['rewD_sum'] = sum(recorder['rew_MD'])
            info['rewN_sum'] = sum(recorder['rew_N'])
            info['rewP_sum'] = sum(recorder['rew_P'])
            info['MD_sum'] = sum(recorder['MD'])
            info['N_sum'] = sum(recorder['N'])
            info['ep_len'] = self.cnt
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
        
    def mutate(self, action):
        if self.mutation:
            return self.sampleRandomVector(32)
        epsilon = 0.01
        new_action = action + epsilon * (np.random.rand((self.nz))*2-1)
        return np.where(np.logical_and(self.action_space.low<=new_action, new_action<=self.action_space.high), new_action, action)
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