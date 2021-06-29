import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
os.chdir(sys.path[0])
from utils.level_process import *
import numpy as np
from root import rootpath


class Identify:
    def __init__(self):
        self.mp = {}
        rule_level = json.load(open(rootpath + "//CNet//data//all_elm_rule.json"))
        for data in rule_level:
            h = data[0]
            for i in range(-2, 3):
                data[0] = h + i
                keyv = tuple(data)
                self.mp[keyv] = 1

    def compare(self, before, after):
        print('compare:',before,after)
        with open(before) as f:
            strlv = f.read()
            before = numpy_level(strlv)
        with open(after) as f:
            strlv = f.read()
            after = numpy_level(strlv)
        U = self.getU(before)
        set1 = self.getWrong(before, pos=U)
        set2 = self.getDif(before, after, pos=U)
        set6 = U - set1
        set4 = set1 & set2
        set3 = set1 - set2
        set5 = set2 - set1
        set3_1 = self.getWrong(after, pos=set3)
        set4_1 = self.getWrong(after, pos=set4)
        set5_1 = self.getWrong(after, pos=set5)
        set6_1 = self.getWrong(after, pos=set6)
        print('W->W:', len(set4_1))
        print('W->T:', len(set4) - len(set4_1))
        print('T->W:', len(set5_1))
        print('T->T:', len(set5) - len(set5_1))
        print('W=W:', len(set3_1))
        print('W=T:', len(set3) - len(set3_1))
        print('T=W:', len(set6_1))
        print('T=T:', len(set6) - len(set6_1))

    def getU(self, lv):
        w, h = len(lv), len(lv[0])
        lv = np.lib.pad(lv, (1, 1), 'constant', constant_values=11)
        res = set()
        for i in range(1, w + 1):
            for j in range(1, h + 1):
                val = (i - 1, lv[i - 1][j - 1], lv[i - 1][j], lv[i - 1][j + 1],
                       lv[i][j - 1], lv[i][j + 1], lv[i + 1][j - 1], lv[i + 1][j], lv[i + 1][j + 1], lv[i][j])
                for k in range(1, 10):
                    if val[k] in [6, 7, 8, 9]:
                        res.add((i - 1, j - 1))
                        break
        return res

    def getWrong(self, lv, pos, ouput=False):
        w, h = len(lv), len(lv[0])
        lv = np.lib.pad(lv, (1, 1), 'constant', constant_values=11)
        res = set()
        for x, y in pos:
            i, j = x + 1, y + 1
            val = (i - 1, lv[i - 1][j - 1], lv[i - 1][j], lv[i - 1][j + 1],
                   lv[i][j - 1], lv[i][j + 1], lv[i + 1][j - 1], lv[i + 1][j], lv[i + 1][j + 1], lv[i][j])
            if val not in self.mp.keys():
                if ouput: print(x, y, val)
                res.add((x, y))
        return res

    def getDif(self, lv1, lv2, pos):
        res = set()
        for i, j in pos:
            if lv1[i][j] != lv2[i][j]:
                res.add((i, j))
        return res


if __name__ == '__main__':
    idf = Identify()
    idf.compare("result//start.txt", "result//result.txt")
