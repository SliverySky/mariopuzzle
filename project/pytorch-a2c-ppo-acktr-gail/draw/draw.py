import matplotlib.pyplot as plt
import csv
import numpy as np
import json
from scipy.interpolate import make_interp_spline
names = ["7"]
color_set=['#ff000a', '#0000ff', '#00ff00','#939393'] #['#ff000a', '#ff9498', '#0000ff', '#8787fd', '#00ff00', '#85f585', '#939393', '#c4c4c4']
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
#plt.tight_layout()

fig = plt.figure()
ax = plt.subplot(111)
box = ax.get_position()
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
def cal():
    with open("/home/cseadmin/sty/project2/test/data.json", "r") as f:
        d = json.load(f)
    a = []
    for i in range(100):
        a.append(d['r'][i])
    
    print(max(a), min(a), np.mean(a), np.std(a))
cal()
def get(s):
    with open("/home/cseadmin/sty/project2/test/data.json", "r") as f:
        d = json.load(f)
    a, b = [], []
    print(len(d['x']))
    for i in range(len(d['x'])):
        a.append(d['x'][i])
        b.append(d[s][i])
    return a,b
for i in range(len(names)):
    exp = names[i]
    exp = "experiment" + exp
    with open('../logs/'+exp+'/result.csv') as f:
        f_csv = csv.reader(f)
        x, y, std, N,MD, p= [], [] ,[], [], [], []
        N_norm, MD_norm = [], []
        first = True
        for row in f_csv:
            if first:
                first = False
                continue
            x.append(float(row[0]))
            y.append(float(row[1]))
            N.append(float(row[6]))
            N_norm.append(float(row[3]))
            MD.append(float(row[5]))
            MD_norm.append(float(row[2]))
            p.append(float(row[4]))
        x, N, MD, p =np.array(x), np.array(N), np.array(MD), np.array(p)
        N_norm, MD_norm =np.array(N_norm), np.array(MD_norm)
        y = np.array(y)
        #print(p.shape,x.shape)
        plt.xlabel("Epoch",font1)
        plt.ylabel("Cumulative reward", font1)
        ap_val=0.8
        r_norm =N_norm+MD_norm+p
        r = N + MD + p
        #plt.plot(x[0:10],r_norm[0:10] , color = color_set[3], alpha=ap_val)
        #a, b = get('_norm')
        #plt.plot(a,[np.mean(np.array(b)) for k in range(len(a))], color_set[1])
        #a, b = get('MD_max')
        #a, c = get('MD_min')
        #plt.plot(a,b)
        plt.plot(x,r)
        #plt.plot(x, variance, color = color_set[1], alpha=ap_val)#,marker='x')
        #plt.plot(x, novelty, color = color_set[2], alpha=ap_val)#,marker='o')
        #plt.plot(x, p,color = color_set[0], alpha=ap_val)#, marker='+')
        #plt.plot(x, z, c="r")
        #plt.fill_between(x, y-std, y+std, facecolor="yellow", alpha=0.5)
plt.legend(["$\pi_{FHP}$"], prop=font2)
#plt.legend(["$\pi_{FHP}$", "$\pi_{random}$"], prop=font2, frameon=False)
#plt.legend(["max (recent 1000)", "min (recent 1000)"], prop=font2, frameon=False)
#plt.legend(["N"], prop=font1)
plt.tick_params(labelsize=15)
#plt.title("Training graph for agents with different reward function")
plt.savefig("norm")
plt.clf()
