import sys,os
import numpy as np
sys.path.append(os.path.dirname(sys.path[0]))
from pcg_gym.envs.MarioLevelRepairer.GA.repair import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
class Repairer():
    def __init__(self, index, gpu = False):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(index)
        net_name = rootpath + "//CNet//dict.pkl"
        if gpu:
            self.net = torch.load(net_name)
            self.net = self.net.cuda()
        else:
            self.net = torch.load(net_name, map_location='cpu')
        self.net.eval()
        self.device = torch.device("cuda:"+str(index) if gpu else "cpu")
    def repair(self,lv):
        return GA(self.net, lv, "", isfigure=False, isrepair=True, device=self.device)
    def cal_draw_defect(self, lv):
        return cal_draw_defect(self.net, lv)
