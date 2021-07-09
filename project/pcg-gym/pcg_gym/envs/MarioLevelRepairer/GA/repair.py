import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
import torch
import torch.nn.functional as F
import copy
from pcg_gym.envs.MarioLevelRepairer.utils.visualization import *
from pcg_gym.envs.MarioLevelRepairer.CNet.model import CNet
from pcg_gym.envs.MarioLevelRepairer.utils.level_process import type_num
from pcg_gym.envs.MarioLevelRepairer.utils.level_process import empty
import copy
from torch.autograd import Variable
import numpy as np
# parameter
Threshold = 0.05
P_M0 = 0.8
P_M1 = 0  # 1/len(S)
RRT_M = 4
Lamda = 20
Iteration = 50
RepairRatio = 0.3
Repeat = 1
origin = None
net = None

S = []
hash_map = {}
repair_set = {}
picture_name = ''
xv = []
score = []  # score[i][j] = {"fit": 0, "value": 0, "replace": 0, "wrong": 0} i means iteration, j means index of individual
bestScore = []
net_device = "cuda:0"


def crossOver(ind1, ind2):
    indvd1 = copy.deepcopy(ind1)
    indvd2 = copy.deepcopy(ind2)
    for i in S:
        if random.random() < 0.5:
            tmp = indvd1[i]
            indvd1[i] = indvd2[i]
            indvd2[i] = tmp
    return indvd1, indvd2

def repairTile(ind):
    pos = []
    for item in S:
        flag, pro_tile = get_protile(ind, item[0], item[1])
        if flag:
            if len(pro_tile) > 0 and ind[item] not in pro_tile:
                pos.append(item)
    random.shuffle(pos)  # randomize the order of repair
    for v in range(len(pos)):
        if random.random() < RepairRatio:
            item = pos[v]
            flag, pro_tile = get_protile(ind, item[0], item[1])
            if flag:
                if len(pro_tile) > 0 and ind[item] not in pro_tile:
                    ind[item] = pro_tile[int(random.random() * len(pro_tile))]

def mutation(ind):
    for item in S:
        if random.random() < P_M1:
            flag, pro_tile = get_protile(ind, item[0], item[1])
            if flag:
                ind[item] = pro_tile[int(random.random() * len(pro_tile))]
            # if random.random() < 0.5:
            #     flag, pro_tile = get_protile(ind, item[0], item[1])
            #     if flag:
            #         ind[item] = pro_tile[int(random.random() * len(pro_tile))]
            # else:
            #     ind[item] = origin[item[0]][item[1]]

def get_step(ind):
    cnt = 0
    for i, j in S:
        if origin[i][j] != ind[(i, j)]:
            cnt += 1
    return cnt

def select(pop, children):
    big = []
    for ind in pop:
        ind['RRT'] = 0
        big.append(ind)
    for ind in children:
        ind['RRT'] = 0
        big.append(ind)
    for i in range(len(big)):
        cnt = 0
        while cnt < RRT_M:
            tmp = int(random.random() * len(big))
            if tmp != i:
                if big[i]['fit'] < big[tmp]['fit']:
                    big[i]['RRT'] += 1
                else:
                    big[tmp]['RRT'] += 1
                cnt += 1
    big.sort(key=lambda x: x['RRT'], reverse=True)
    big = big[:Lamda]
    return big


def inmap(i, j):
    if 0 <= i < len(origin) and 0 <= j < len(origin[0]):
        return True
    else:
        return False

def update_probility(pop):
    pop.sort(key=lambda x: x['fit'], reverse = True)
    total = (1 + len(pop)) * len(pop) / 2
    for i in range(len(pop)):
        pop[i]['p'] = (i + 1) / total

def random_choose_ind(pop):
    x = random.random()
    cnt = 0
    for i in range(len(pop)):
        cnt += pop[i]['p']
        if cnt > x:
            return i
    print('not find:', cnt, x)
    return len(pop) - 1

def evolution(isfigure, isrepair, result_path):
    figure_index = range(Iteration)  # [0, 1, 2, 4, 8, 15, 25, 40, 60, 90]
    figure_path = result_path + "//figure"
    txt_path = result_path + "//txt"
    global S
    S = []
    initial()
    pop = initpop()
    best = pop[0]
    start = {}
    for i, j in S:
        start[(i, j)] = origin[i][j]
    level, S1, T1 = get_mark_set(start)
    if isfigure:
        saveLevelAsImage(level, result_path + "//start")
        saveAndMark(level, result_path + "//start(Remark)", T1, S1)
        saveAndMark(level, figure_path + "//iteration0", T1, S1)
    #save_level_as_text(level, txt_path + "//iteration0")
    #save_level_as_text(level, result_path + "//start")
    for index in range(Iteration):
        level, S1, T1 = get_mark_set(best)
        #save_level_as_text(level, txt_path + "//iteration" + str(index + 1))
        if (index in figure_index) and isfigure:
            saveAndMark(level, figure_path + "//iteration" + str(index + 1), T1, S1)
        for ind in pop:
            update_fitness(ind)
        for i in range(Lamda):
            for j in score[index][i].keys():
                score[index][i][j] += pop[i][j]
        update_probility(pop)
        children = []
        while len(children) < Lamda:
            x, y = None, None
            while True:
                x, y = random_choose_ind(pop), random_choose_ind(pop)
                if x != y: break
            x1, y1 = crossOver(pop[x], pop[y])
            children.append(x1)
            children.append(y1)

        # update score

        for ind in children:
            if random.random() < P_M0:
                mutation(ind)
        if isrepair:
            for ind in children:
                repairTile(ind)
        for ind in children:
            update_fitness(ind)
        pop = select(pop, children)
        for ind in pop:
            if ind['fit'] < best['fit']:
                best = copy.deepcopy(ind)
        xv.append(index)
        avg = 0
        for ind in pop:
            avg += ind['fit']
        #print("iter=", index, "best_fit=", best['fit'], 'avg=', avg / Lamda)
    '''for i in range(Lamda):
        for j in score[Iteration].keys():
            score[Iteration][j] += pop[i][j]'''
    level, S1, T1 = get_mark_set(best)
    if isfigure:
        saveLevelAsImage(level, result_path + "//result")
        saveAndMark(level, result_path + "//result(Remark)", T1, S1)
    return level

def get_protile(ind, i, j):
    flag = False
    condition = [i]
    # condition.append(i)
    height = len(origin)
    width = len(origin[0])
    for offset in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]:
        ni = i + offset[0]
        nj = j + offset[1]
        if ni < 0 or nj < 0 or ni >= height or nj >= width:
            condition.append(empty)
        else:
            if (ni, nj) in ind.keys():
                tmp = ind[(ni, nj)]
            else:
                tmp = origin[ni][nj]
            if 6 <= tmp <= 9:
                flag = True
            if ni == i and nj == j:
                continue
            condition.append(tmp)
    if flag:
        if tuple(condition) in hash_map.keys():
            pro_tile = hash_map[tuple(condition)]
        else:
            if (i, j) in ind.keys():
                y = ind[(i, j)]
            else:
                y = origin[i][j]
            x = torch.zeros(8*type_num + 1)
            x[0] = condition[0]
            for i1 in range(1, 9):
                x[(i1-1) * type_num + 1 + condition[i1]] = 1
            with torch.no_grad():
                if str(net_device) != 'cpu':
                    new = net(Variable(x).cuda())
                else:
                    new = net(Variable(x))
                pro = F.softmax(new, dim=0)
            pro_tile = []
            pro_num = []
            for i1 in range(type_num-1):
                if pro[i1] >= Threshold:
                    pro_tile.append(i1)
                    pro_num.append(pro[i1])
            hash_map[tuple(condition)] = pro_tile
        return True, pro_tile
    else:
        return False, []

def fitness_fuction(pro_sum, error_sum, step):
    return pro_sum + 5 * error_sum + 3 * step

def initial():
    global P_M1
    height = len(origin)
    width = len(origin[0])
    for i in range(height):
        for j in range(width):
            flag, pro_tile = get_protile({}, i, j)
            if flag:
                if len(pro_tile) > 1 or origin[i][j] not in pro_tile:
                    S.append((i, j))
                    # print(i,j)
    #print("#S=", len(S))
    if(len(S)>0):
        P_M1 = 1 / len(S)

def initpop():
    pop = []
    pos = []
    for i in range(Lamda):
        ind = {}
        for item in S:
            ind[item] = origin[item[0]][item[1]]
        pop.append(ind)
    for item in S:
        flag, pro_tile = get_protile({}, item[0], item[1])
        if flag and origin[item[0]][item[1]] in pro_tile:
            pos.append(item)
    for i in range(Lamda):
        random.shuffle(pos)
        for item in pos:
            flag, pro_tile = get_protile(pop[i], item[0], item[1])
            if flag and pop[i][item] in pro_tile:
                pop[i][item] = pro_tile[int(random.random() * len(pro_tile))]
        repairTile(pop[i])
    return pop

def update_fitness(ind):
    pro_sum = 0
    error_sum = 0

    for i, j in S:
        flag, pro_tile = get_protile(ind, i, j)
        if flag:
            if (ind[(i, j)] in pro_tile) and len(pro_tile) > 1:
                pro_sum += len(pro_tile)
            elif ind[(i, j)] not in pro_tile:
                error_sum += 1
                pro_sum += len(pro_tile)

    ind['wrong'] = error_sum
    ind['value'] = pro_sum
    ind['replace'] = get_step(ind)
    ind['fit'] = fitness_fuction(pro_sum, error_sum, ind['replace'])

def get_mark_set(ind):
    S1 = []
    T1 = []
    level = copy.deepcopy(origin)
    for i, j in S:
        level[i][j] = ind[(i, j)]
    for i, j in S:
        flag, pro_tile = get_protile(ind, i, j)
        if flag:
            if (ind[(i, j)] in pro_tile) and len(pro_tile) > 1:
                S1.append((i, j))
            elif ind[(i, j)] not in pro_tile:
                T1.append((i, j))
    return level, S1, T1

def GA(_net, lv, result_path="", isfigure=True, isrepair=True, device="cpu"):
    global origin
    global net
    global score
    global net_device
    net = _net
    score = []
    net_device = device
    for i in range(Iteration):
        one = []
        for j in range(Lamda):
            one.append({"fit": 0, "value": 0, "replace": 0, "wrong": 0})
        score.append(one)
    whole_level = copy.deepcopy(lv)
    origin = whole_level
    return evolution(isfigure, isrepair, result_path)
    #with open(result_path + "//json//data.json", 'w') as f:
    #    json.dump(score, f)
def cal_draw_defect(_net, lv):
    global origin
    global net
    global score
    global S
    net = _net
    score = []
    whole_level = copy.deepcopy(lv)
    origin = whole_level
    S = []
    initial()
    pop = initpop()
    best = pop[0]
    start = {}
    for i, j in S:
        start[(i, j)] = origin[i][j]
    level, S1, T1 = get_mark_set(start)
    return (T1,S1)
