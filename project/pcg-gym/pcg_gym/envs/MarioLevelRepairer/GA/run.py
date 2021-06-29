import sys,os
import numpy as np
sys.path.append(os.path.dirname(sys.path[0]))
from GA.repair import *
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',default='lv1', help='level name')
    args = parser.parse_args()
    
    net_name = rootpath + "//CNet//dict.pkl"
    lv_name = rootpath + "//LevelGenerator//RL//"+args.name+".txt"
    result_path = rootpath + "//GA//result//"+args.name
    lv = ""
    #print('repair lv:',lv_name)
    #print('used CNet:',net_name)
    #print('saved path:',result_path)
    net = torch.load(net_name).to("cpu")
    net.eval()
    for e in open(lv_name).readlines():
        lv = lv + e
    lv = numpy_level(lv)
    for i in range(100):
        GA(net, lv_name, result_path, isfigure=False, isrepair=True)
