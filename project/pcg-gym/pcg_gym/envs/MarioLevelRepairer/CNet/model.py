import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
os.chdir(sys.path[0])
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json
import pcg_gym.envs.MarioLevelRepairer.utils.level_process
import pcg_gym.envs.MarioLevelRepairer.utils as utils
type_num = utils.level_process.type_num
empty = utils.level_process.empty
class CNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_hidden2, n_output):
        super(CNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
        self.predict = torch.nn.Linear(n_hidden2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)
        return x


if __name__ == '__main__':
    rule_level = json.load(open("./data/legal_rule.json"))
    batch_size = 8
    total = 20000
    USEGPU = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and USEGPU else "cpu")
    gpus = [0]
    cnet_num = 1
    siz = type_num * 8 + 1
    for t in range(cnet_num):
        net = CNet(siz, 200, 100, type_num)
        net = net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
        loss_func = torch.nn.CrossEntropyLoss()
        data1 = []
        data2 = []
        for i in range(len(rule_level)):
            val1 = [0.0] * siz
            val1[0] = rule_level[i][0]
            for k in range(1, 9):
                val1[(k-1) * type_num +1 + rule_level[i][k]] = 1
            val2 = rule_level[i][9]
            data1.append(val1)
            data2.append(val2)
        data1 = np.array(data1)
        data2 = np.array(data2)
        for i in range(total):
            perm = torch.randperm(len(data1))
            data1 = data1[perm]
            data2 = data2[perm]
            sum = 0
            for j in range(len(rule_level) // batch_size):
                input = Variable(torch.tensor(data1[batch_size * j:batch_size * (j + 1)]).float(), requires_grad=True)
                input = input.to(device)
                target = Variable(torch.LongTensor(data2[batch_size * j:batch_size * (j + 1)]))
                target = target.to(device)
                optimizer.zero_grad()
                input = net(input)
                loss = loss_func(input, target)
                loss.backward()
                optimizer.step()
                if USEGPU:
                    sum += loss.cpu().detach().numpy()
                else:
                    sum += loss.detach().numpy()
            print("\rNet", t + 1, "(size=", batch_size, ")iter=", i+1, "/", total, end='')
            print("     loss=", sum)
        torch.save(net, "dict.pkl")
