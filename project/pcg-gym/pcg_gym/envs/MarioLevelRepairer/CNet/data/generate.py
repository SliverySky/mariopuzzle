import sys,os
sys.path.append(sys.path[0]+'/..'+'/..')
os.chdir(sys.path[0])
import random
from utils.level_process import *
import json
def random_change(data, change_n):
    res = data.copy()
    for i in range(change_n):
        res[int(random.random()*8)+1]=int(random.random()*11)
    return res
def generate_illegal():
    rule_level = json.load(open("legal_rule.json"))
    rule_set = {}
    cnt = [0] * 11
    for e in rule_level:
        data = tuple(e[0:9])
        if data not in rule_set.keys():
            rule_set[data]=set()
        rule_set[data].add(e[9])
    fake_level = []
    for e in rule_set.keys():
        for i in range(11):
            if i not in rule_set[e]:
                fake_level.append(list(e) + [i])
                cnt[i] += 1
    path = 'illegal_rule.json'
    with open(path, "w") as f:
        json.dump(fake_level, f)
        print('generate ',path)
def generate_fake(name, change_n):
    rule_level = json.load(open(name+".json"))
    rule_set = set()
    res = []
    for i in rule_level:
        data = tuple(i[0:9])
        rule_set.add(data)
    for i in rule_level:
        data_f = random_change(i[0:9], change_n)
        while tuple(data_f) in rule_set:
            data_f = random_change(i[0:9], change_n)
        res.append(data_f + [i[9]])
    path = name+'_F' + str(change_n) + '.json'
    with open(path, "w") as f:
        json.dump(res, f)
        print('generate ',path)
if __name__ == '__main__':
    getRuleData() # generate legal_rule.json
    generate_illegal() # generate illegal_rule.json
    getAllElmRuleData()
    for i in range(1,4):
        generate_fake('legal_rule', i)
        generate_fake('illegal_rule', i)
