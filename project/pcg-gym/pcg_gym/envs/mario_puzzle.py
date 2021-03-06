import socket
import subprocess
from subprocess import *
from atexit import register
import os
import sys
import time
import gym
from gym import spaces
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
import torch
import json
import random
import numpy as np
import math
import os
from collections import deque
import subprocess
import sys
import copy
from pcg_gym.envs.generator2 import Generator
from pcg_gym.envs.utils import *
from pcg_gym.envs.MarioLevelRepairer.GA.repairer import Repairer
rootpath = os.path.abspath(os.path.dirname(__file__))


def clean(prog):
    try:
        prog.close()
    except:
        pass


class AStarAgent():
    '''
    Comunicate with a java program located in  pcg_gym/envs/Mario-AI-Framework-master.jar
    The program contains a mario game with an A* agent to test the generated segments.
    '''

    def __init__(self, id, visuals):
        self.id = id
        self.test_id = 0
        self.visuals = '1' if visuals else '0'

    def start(self):
        register(clean, self)
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # find an unused port
        while True:
            self.port = int(random.random()*5000)+2000
            try:
                self.client.connect(('localhost', self.port))
                self.client.shutdown(2)
            except:
                break

        os.system("java -jar "+rootpath + "/Mario-AI-Framework-master.jar " + rootpath +
                  "/test_pool/" + str(self.id) + " " + str(self.port)+" "+self.visuals+" &")
        while True:
            try:
                self.client.connect(('localhost', self.port))
                break
            except:
                pass

    def close(self):
        msg = 'close'
        self.client.send(msg.encode('utf-8'))
        self.client.close()
        self.test_id = 0

    def start_test(self, lv):
        '''
        Test the first segment. The initial position of Mario is set as default
        '''
        return self.__test(lv, "start\n")

    def continue_test(self, lv):
        '''
        Used when the previous segment is playable.
        The initial postion of Mario is set according to previous test. 
        '''
        return self.__test(lv, "continue\n")

    def retest(self, lv):
        '''
        Used when the previous segment is unplayable.
        The initial postion of Mario keeps the same as the previous test. 
        '''
        return self.__test(lv, "retest\n")

    def __test(self, lv, msg):
        name = str(self.id)+"_"+str(self.test_id)
        saveLevelAsText(lv, rootpath+"/test_pool/"+name)
        res = self.client.send(msg.encode('utf-8'))
        data = self.client.recv(1024)
        rate = float(data.decode())
        os.remove(rootpath+"/test_pool/"+name+".txt")
        self.test_id += 1
        return rate


class MarioPuzzle(gym.Env):
    def __init__(self):
        self.id = random.randint(1, 10000000)
        self.nz = 32
        self.elm_num = 13
        self.action_space = spaces.Box(-1., 1.,
                                       shape=(self.nz,), dtype='float32')
        self.observation_space = spaces.Box(-1.,
                                            1., shape=(self.nz,), dtype='float32')
        self.F_que = deque(maxlen=1000)
        self.H_que = deque(maxlen=1000)
        self.use_P, self.use_F, self.use_H = False, False, False
        self.pop = deque(maxlen=20)
        self.novel_k = 10
        self.initial_state = None
        self.recorder = {}  # record information
        self.start_time = 0
        self.online = False
        self.skip = False
        self.last_play = False  # whether last segment is playable

    def kl_fn(self, val):
        if (val < 0.26):
            return -(val-0.26)**2
        if (val > 0.94):
            return -(val-0.94)**2
        return 0

    def setParameter(self, info, index=0):
        exp = info['exp']
        visuals = 'visuals' in info.keys()  # render the generated level segments
        self.skip = 'skip' in info.keys()
        cuda = 'cuda' in info.keys() and info['cuda']
        cuda_num = len(info['cuda_id']) if cuda else 1
        cuda_id = info['cuda_id'][index % cuda_num] if cuda else 0
        self.cnt = 0
        self.map_h = 14
        self.map_w = 14*100
        self.win_h, self.win_w = 14, 14
        self.sy, self.sx = 14, 7
        self.ny, self.nx = 0, 3
        '''
        set reward function
        F: fun
        H: historical deviation
        P: playability
        You can also design you own reward function
        '''
        if exp == 1:  # F
            self.use_F, self.use_H, self.use_P = True, False, False
        elif exp == 2:  # H
            self.use_F, self.use_H, self.use_P = False, True, False
        elif exp == 3:  # P
            self.use_F, self.use_H, self.use_P = False, False, True
        elif exp == 4:  # FH
            self.use_F, self.use_H, self.use_P = True, True, False
        elif exp == 5:  # FP
            self.use_F, self.use_H, self.use_P = True, False, True
        elif exp == 6:  # HP
            self.use_F, self.use_H, self.use_P = False, True, True
        elif exp == 7:  # FHP
            self.use_F, self.use_H, self.use_P = True, True, True

        self.norm = (exp > 3)
        self.agent = AStarAgent(self.id, visuals)
        self.agent.start()
        model_path = os.path.dirname(
            __file__) + "//models//" + str(self.win_h)+"_"+str(self.win_w)+".pth"
        # A generator to generate level segments
        self.generator = Generator(self.id, cuda_id, cuda)
        # A repairer to repair the broken pipes
        self.repairer = Repairer(cuda_id, cuda)
        self.tot = (self.map_h // self.win_h) * (self.map_w //
                                                 self.win_w)  # the maximum game length

    def sampleRandomVector(self, size):
        return np.clip(np.random.randn(size), self.action_space.low, self.action_space.high)

    def add_then_norm(self, value, history):
        if not self.norm:
            return value
        history.append(value)
        maxv = max(history)
        minv = min(history)
        if maxv == minv:
            return 0
        else:
            return (value-minv)/(maxv-minv)

    def reset(self):
        recorder = self.recorder = {}
        recorder['defect'] = []  # the number of defects for each segment
        recorder['sample'] = []  # the resample time for each segment
        recorder['time'] = []  # the time needed to generate each segment
        recorder['D'] = []  # diversity
        recorder['H'] = []  # historical deviation (novelty)
        recorder['F'] = []  # fun (moderate diversity)
        recorder['rew_F'] = []  # reward for fun (normalized if needed)
        recorder['rew_H'] = []  # reward for historical deviation (normalized if needed)
        recorder['rew_P'] = []  # reward for playability
        self.start_time = time.time()
        self.lv = np.zeros(shape=(self.map_h, self.map_w), dtype=np.uint8)
        self.unrepair_lv = np.zeros(
            shape=(self.map_h, self.map_w), dtype=np.uint8)
        playable = False
        # sample a playable initial state randomly
        while not playable:
            if self.initial_state != None:
                self.state = self.initial_state
            else:
                self.state = self.sampleRandomVector(self.nz)
            st = time.time()
            piece = self.generator.generate(self.state)  # generate the segment
            st = time.time()
            self.unrepair_lv[0:self.win_h, 0:self.win_w] = piece
            new_piece = self.repairer.repair(piece)  # repair the segment
            # test the segment by A* agent
            rate = self.agent.start_test(new_piece)
            if rate == 1.0:
                recorder['sample'].append(1)
                playable = True
            else:
                recorder['sample'].append(0)
            recorder['time'].append(time.time()-self.start_time)
            self.start_time = time.time()
        # fill the whole level with the repaired segment
        self.lv[0:self.win_h, 0:self.win_w] = new_piece
        self.cnt = 1
        self.last_play = True
        self.pop.clear()
        self.pop.append(lv2Map(self.lv[0:self.win_h, 0:self.win_w]))
        return self.state

    def step(self, action):
        if self.online:
            return self.online_test(action)
        recorder = self.recorder
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state = action  # update the state
        now_x = (self.cnt // (self.map_h // self.win_h)) * self.win_w
        piece = self.generator.generate(self.state)
        self.lv[:, now_x: now_x+self.win_w] = piece
        self.unrepair_lv[:, now_x: now_x+self.win_w] = piece
        # repair two adjacent level segments both.
        self.lv[:, now_x-self.win_w: now_x+self.win_w] = self.repairer.repair(
            self.lv[:, now_x-self.win_w: now_x+self.win_w])
        reward, done = 0, False
        # calculate playability
        if self.use_P:
            rate = self.agent.continue_test(
                self.lv[:, max(0, now_x-3*self.win_w): now_x+self.win_w])
            reward += rate
            if rate < 1.0:  # if the segment is unplayable, the game ends
                done = True
            recorder['rew_P'].append(rate)
        # calculate the diversity
        kl_val = KLWithSlideWindow(
            self.lv, (0, now_x, self.win_h, self.win_w), self.sx, self.nx, self.sy, self.ny)
        # calculate fun 
        if self.use_F:
            recorder['D'].append(kl_val)
            recorder['F'].append(self.kl_fn(kl_val))
            rew_F = self.add_then_norm(self.kl_fn(kl_val), self.F_que)
            recorder['rew_F'].append(rew_F)
            reward += rew_F
        # calculate historical deviation
        if self.use_H:
            piece_mp = lv2Map(self.lv[:, now_x: now_x + self.win_w])
            novelty = self.cal_novelty(piece_mp)
            recorder['H'].append(novelty)
            rew_H = self.add_then_norm(novelty, self.H_que)
            recorder['rew_H'].append(rew_H)
            reward += rew_H
            self.pop.append(piece_mp)
        self.cnt += 1
        if self.cnt >= self.tot:
            done = True
        info = {}
        if done:
            info['rewF_sum'] = sum(recorder['rew_F'])
            info['rewH_sum'] = sum(recorder['rew_H'])
            info['rewP_sum'] = sum(recorder['rew_P'])
            info['F_sum'] = sum(recorder['F'])
            info['H_sum'] = sum(recorder['H'])
            info['ep_len'] = self.cnt
            self.recorder['lv'] = self.lv[:, 0:now_x+self.win_w]
            self.recorder['unrepair_lv'] = self.unrepair_lv[:,
                                                            0:now_x+self.win_w]
            info['recorder'] = recorder
        return self.state, reward, done, info

    '''
    online generate playable levels. 
    A* agent is used for online test (may be surrogate model in the future)
    '''

    def online_test(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        recorder = self.recorder
        self.start_time = time.time()
        info = {}
        now_x = (self.cnt // (self.map_h // self.win_h)) * self.win_w
        if not self.last_play:
            self.lv = self.last_lv
        self.last_lv = copy.deepcopy(self.lv)
        piece = self.generator.generate(action)
        self.lv[:, now_x: now_x+self.win_w] = piece
        self.unrepair_lv[:, now_x: now_x+self.win_w] = piece
        # repair two adjacent level segments both.
        self.lv[:, now_x-self.win_w: now_x+self.win_w] = self.repairer.repair(
            self.lv[:, now_x-self.win_w: now_x+self.win_w])
        # use A* agent to test the playability
        if self.last_play:
            rate = self.agent.continue_test(
                self.lv[:, max(0, now_x-3*self.win_w):now_x+self.win_w])
        else:
            rate = self.agent.retest(
                self.lv[:, max(0, now_x-3*self.win_w):now_x+self.win_w])
        if rate == 1:
            self.cnt += 1
            self.last_play = True
            recorder['sample'].append(1)
            self.state = action
        else:
            self.last_play = False
            recorder['sample'].append(0)
        recorder['time'].append(time.time()-self.start_time)
        done = False
        info['playable'] = self.last_play
        resample_cnt = 0
        for i in range(len(recorder['sample'])):
            if recorder['sample'][len(recorder['sample'])-1-i] == 0:
                resample_cnt += 1
            else:
                break
        # if maximum game length is reached (success) or resample time >= 20 (fail), the game ends
        if self.cnt >= self.tot or resample_cnt >= 20:
            recorder['defect_before'] = self.repairer.cal_draw_defect(
                self.unrepair_lv)
            recorder['defect_end'] = self.repairer.cal_draw_defect(self.lv)
            info['recorder'] = recorder
            info['lv'] = self.lv
            info['unrepair_lv'] = self.unrepair_lv
            done = True
        return self.state, 0, done, info

    def cal_novelty(self, piece):
        score = []
        for x in self.pop:
            score.append(calKLFromMap(x, piece))
        score.sort()
        sum = 0
        siz = min(len(score), self.novel_k)
        for i in range(siz):
            sum += score[i]
        if siz > 0:
            sum /= siz
        return sum
